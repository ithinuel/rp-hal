use std::env;
use std::path::PathBuf;

use compiletest_rs as compiletest;

fn run_mode(mode: &'static str, custom_dir: Option<&'static str>) {
    let mut config = compiletest::Config::default();
    let cfg_mode = mode.parse().expect("Invalid mode");

    config.target = "thumbv6m-none-eabi".to_string();
    config.mode = cfg_mode;

    let dir = custom_dir.unwrap_or(mode);
    config.src_base = PathBuf::from(format!("tests/{}", dir));
    config.target_rustcflags = Some(
            "-L target/thumbv6m-none-eabi/debug
            -L target/thumbv6m-none-eabi/debug/deps".to_string());
    config.llvm_filecheck = Some(
        env::var("FILECHECK")
            .unwrap_or("FileCheck".to_string())
            .into(),
    );
    
    config.clean_rmeta();
    config.strict_headers = true;
    config.edition = Some("2021".to_string());
    config.verbose = true;

    compiletest::run_tests(&config);
}

#[test]
fn compile_test() {
    run_mode("compile-fail", None);
    //run_mode("ui", None);
    //run_mode("ui", Some("nightly"));
}
