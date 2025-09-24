use assert_cmd::Command;

#[tokio::test]
async fn validate_real_world_dataset() {
    let mut cmd = Command::cargo_bin("pensieve-validator").expect("binary exists");
    cmd.arg("validate")
        .arg("--directory")
        .arg("/Users/neetipatni/downloads/RustRAW20250920")
        .arg("--confirm")
        .arg("--output-dir")
        .arg("./validation_reports")
        .assert()
        .success();
}