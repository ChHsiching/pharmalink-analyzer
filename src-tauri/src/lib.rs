use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            #[cfg(not(debug_assertions))]
            {
                spawn_backend_sidecar(app.handle().clone())?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(not(debug_assertions))]
fn spawn_backend_sidecar(app: tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let sidecar_command = app.shell().sidecar("binaries/pharmalink-backend")?;

    let (mut rx, child) = sidecar_command.spawn()?;

    app.manage(BackendChild(child));

    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let CommandEvent::Stdout(line_bytes) = event {
                let line = String::from_utf8_lossy(&line_bytes);
                println!("[backend] {}", line);
            } else if let CommandEvent::Stderr(line_bytes) = event {
                let line = String::from_utf8_lossy(&line_bytes);
                eprintln!("[backend] {}", line);
            }
        }
    });

    Ok(())
}

#[cfg(not(debug_assertions))]
struct BackendChild(tauri_plugin_shell::process::CommandChild);

#[cfg(not(debug_assertions))]
impl Drop for BackendChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}
