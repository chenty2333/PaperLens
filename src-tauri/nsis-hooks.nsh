!macro NSIS_HOOK_POSTUNINSTALL
  MessageBox MB_YESNO|MB_ICONQUESTION "是否同时删除 PaperLens 本机设置和 WebView 缓存？$\r$\n$\r$\n这不会删除你在 PaperLens 里选择的论文库/输出目录。" IDNO done
  RMDir /r "$LOCALAPPDATA\app.paperlens.desktop"
done:
!macroend
