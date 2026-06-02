use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::Duration,
};

use serde::Serialize;
use tauri::{Emitter, Manager};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Default)]
struct CoreServiceState {
    service: Mutex<Option<CoreService>>,
}

struct CoreService {
    child: Arc<Mutex<Child>>,
    base_url: String,
    token: String,
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct CleanupReport {
    removed: Vec<String>,
    missing: Vec<String>,
    errors: Vec<String>,
}

impl CoreService {
    fn request_shutdown(&self) {
        let Some(authority) = self
            .base_url
            .strip_prefix("http://")
            .and_then(|value| value.split('/').next())
        else {
            return;
        };
        let Ok(mut stream) = TcpStream::connect(authority) else {
            return;
        };
        let _ = stream.set_write_timeout(Some(Duration::from_millis(800)));
        let request = format!(
            "POST /shutdown HTTP/1.1\r\nHost: {authority}\r\nX-PaperLens-Token: {}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}",
            self.token
        );
        let _ = stream.write_all(request.as_bytes());
        let _ = stream.flush();
    }
}

impl Drop for CoreService {
    fn drop(&mut self) {
        self.request_shutdown();
        if let Ok(mut child) = self.child.lock() {
            for _ in 0..20 {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => thread::sleep(Duration::from_millis(100)),
                    Err(_) => break,
                }
            }
            let _ = child.kill();
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ServiceInfo {
    base_url: String,
    token: String,
}

fn candidate_dirs(app: &tauri::AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
            dirs.push(parent.join("sidecars"));
        }
    }
    if let Ok(resource_dir) = app.path().resource_dir() {
        dirs.push(resource_dir);
    }
    if let Ok(current_dir) = std::env::current_dir() {
        dirs.push(current_dir.join("src-tauri").join("binaries"));
        dirs.push(current_dir);
    }
    dirs
}

fn find_config_path() -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let path = parent.join("config").join("default_config.json");
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }
    }
    if let Ok(current_dir) = std::env::current_dir() {
        let path = current_dir.join("config").join("default_config.json");
        if path.exists() {
            return path.to_string_lossy().to_string();
        }
    }
    "config/default_config.json".to_string()
}

fn looks_like_core_sidecar(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("paperlens-core"))
        .unwrap_or(false)
}

fn find_core_sidecar(app: &tauri::AppHandle) -> Option<PathBuf> {
    for dir in candidate_dirs(app) {
        if !dir.exists() {
            continue;
        }
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && looks_like_core_sidecar(&path) {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn hide_core_window(command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn remove_cleanup_path(report: &mut CleanupReport, path: PathBuf) {
    if !path.exists() {
        report.missing.push(path_label(&path));
        return;
    }

    let result = if path.is_dir() {
        fs::remove_dir_all(&path)
    } else {
        fs::remove_file(&path)
    };

    match result {
        Ok(()) => report.removed.push(path_label(&path)),
        Err(err) => report
            .errors
            .push(format!("{}: {err}", path_label(&path))),
    }
}

fn add_app_path(
    report: &mut CleanupReport,
    path: Result<PathBuf, tauri::Error>,
    children: &[&str],
) {
    let Ok(root) = path else {
        return;
    };
    if children.is_empty() {
        remove_cleanup_path(report, root);
        return;
    }
    for child in children {
        remove_cleanup_path(report, root.join(child));
    }
}

fn workspace_artifacts(output_dir: &Path) -> Vec<PathBuf> {
    [
        ".paperlens",
        "papers",
        "PaperLens.md",
        "PaperLens.json",
        "PaperLens_Library.md",
        "PaperLens_Library.json",
    ]
    .into_iter()
    .map(|name| output_dir.join(name))
    .collect()
}

fn build_core_command(app: &tauri::AppHandle, args: &[String]) -> Command {
    if let Some(sidecar) = find_core_sidecar(app) {
        let mut command = Command::new(sidecar);
        command.args(args);
        hide_core_window(&mut command);
        return command;
    }

    let mut command = Command::new("python3");
    command.arg("-m").arg("paperlens_core.main").args(args);
    hide_core_window(&mut command);
    command
}

fn existing_service_info(service: &CoreService) -> Option<ServiceInfo> {
    let mut child = service.child.lock().ok()?;
    match child.try_wait() {
        Ok(None) => Some(ServiceInfo {
            base_url: service.base_url.clone(),
            token: service.token.clone(),
        }),
        Ok(Some(_)) | Err(_) => None,
    }
}

fn parse_server_started(line: &str) -> Result<ServiceInfo, String> {
    let value: serde_json::Value = serde_json::from_str(line).map_err(|err| err.to_string())?;
    if value.get("type").and_then(|value| value.as_str()) != Some("server_started") {
        return Err(format!(
            "Unexpected PaperLens Core service startup line: {line}"
        ));
    }
    let base_url = value
        .get("base_url")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "Core service did not report base_url".to_string())?
        .to_string();
    let token = value
        .get("token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "Core service did not report token".to_string())?
        .to_string();
    Ok(ServiceInfo { base_url, token })
}

fn start_core_service(app: tauri::AppHandle) -> Result<CoreService, String> {
    let args = vec![
        "serve".to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        "0".to_string(),
        "--config".to_string(),
        find_config_path(),
    ];
    let mut command = build_core_command(&app, &args);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|err| err.to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Core service stdout is unavailable".to_string())?;
    let stderr = child.stderr.take();

    if let Some(stderr) = stderr {
        let app_stderr = app.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_stderr.emit("core-service-error", line);
            }
        });
    }

    let (startup_tx, startup_rx) = mpsc::channel();
    let app_stdout = app.clone();
    thread::spawn(move || {
        let mut stdout_reader = BufReader::new(stdout);
        let mut first_line = String::new();
        let startup = match stdout_reader.read_line(&mut first_line) {
            Ok(0) => Err("Core service exited before reporting startup".to_string()),
            Ok(_) => parse_server_started(first_line.trim()),
            Err(err) => Err(err.to_string()),
        };
        if startup.is_err() && !first_line.trim().is_empty() {
            let _ = app_stdout.emit("core-service-log", first_line.trim().to_string());
        }
        let _ = startup_tx.send(startup);
        for line in stdout_reader.lines().map_while(Result::ok) {
            let _ = app_stdout.emit("core-service-log", line);
        }
    });

    let info = match startup_rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok(info)) => info,
        Ok(Err(err)) => {
            let _ = child.kill();
            return Err(err);
        }
        Err(_) => {
            let _ = child.kill();
            return Err("Timed out waiting for PaperLens Core service to start".to_string());
        }
    };

    Ok(CoreService {
        child: Arc::new(Mutex::new(child)),
        base_url: info.base_url,
        token: info.token,
    })
}

#[tauri::command]
fn ensure_core_service(
    app: tauri::AppHandle,
    state: tauri::State<'_, CoreServiceState>,
) -> Result<ServiceInfo, String> {
    let mut slot = state
        .service
        .lock()
        .map_err(|_| "Core service state lock poisoned".to_string())?;
    if let Some(service) = slot.as_ref() {
        if let Some(info) = existing_service_info(service) {
            return Ok(info);
        }
    }
    *slot = None;
    let service = start_core_service(app)?;
    let info = ServiceInfo {
        base_url: service.base_url.clone(),
        token: service.token.clone(),
    };
    *slot = Some(service);
    Ok(info)
}

#[tauri::command]
fn shutdown_core_service(state: tauri::State<'_, CoreServiceState>) -> Result<(), String> {
    let mut slot = state
        .service
        .lock()
        .map_err(|_| "Core service state lock poisoned".to_string())?;
    *slot = None;
    Ok(())
}

#[tauri::command]
fn clear_local_app_data(app: tauri::AppHandle) -> CleanupReport {
    let mut report = CleanupReport::default();
    add_app_path(&mut report, app.path().app_log_dir(), &[]);
    add_app_path(
        &mut report,
        app.path().app_local_data_dir(),
        &[
            "EBWebView\\Crashpad",
            "EBWebView\\Default\\Cache",
            "EBWebView\\Default\\Code Cache",
            "EBWebView\\Default\\GPUCache",
            "EBWebView\\Default\\blob_storage",
            "EBWebView\\Default\\Session Storage",
            "EBWebView\\Default\\Sessions",
        ],
    );
    report
}

#[tauri::command]
fn clear_workspace_data(output_dir: String) -> Result<CleanupReport, String> {
    let root = PathBuf::from(output_dir);
    if !root.exists() {
        return Err("当前输出目录不存在。".to_string());
    }
    if !root.is_dir() {
        return Err("当前输出路径不是目录。".to_string());
    }

    let mut report = CleanupReport::default();
    for path in workspace_artifacts(&root) {
        remove_cleanup_path(&mut report, path);
    }
    Ok(report)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(CoreServiceState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            ensure_core_service,
            shutdown_core_service,
            clear_local_app_data,
            clear_workspace_data
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
