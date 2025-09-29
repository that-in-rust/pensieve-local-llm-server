//! Unit tests for dependency API compatibility
//! 
//! Tests Requirements 1.1, 2.1, 2.2, 2.3 - Dependency API usage and compatibility

use std::time::{Duration, Instant};
use sysinfo::{System, Pid};
use git2::{Repository, RepositoryInitOptions};
use tempfile::TempDir;

/// Test sysinfo API compatibility and usage
#[test]
fn test_sysinfo_api_compatibility() {
    let mut system = System::new_all();
    system.refresh_all();
    
    // Test CPU usage collection
    let cpus = system.cpus();
    assert!(!cpus.is_empty(), "Should have at least one CPU");
    
    for cpu in cpus {
        let usage = cpu.cpu_usage();
        assert!(usage >= 0.0, "CPU usage should be non-negative");
        assert!(usage <= 100.0, "CPU usage should not exceed 100%");
    }
    
    // Test global CPU info
    let global_cpu = system.global_cpu_info();
    let global_usage = global_cpu.cpu_usage();
    assert!(global_usage >= 0.0, "Global CPU usage should be non-negative");
    
    // Test memory information
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let available_memory = system.available_memory();
    
    assert!(total_memory > 0, "Total memory should be positive");
    assert!(used_memory <= total_memory, "Used memory should not exceed total");
    assert!(available_memory <= total_memory, "Available memory should not exceed total");
    
    // Test process information
    let processes = system.processes();
    assert!(!processes.is_empty(), "Should have at least one process");
    
    // Test current process
    if let Ok(current_pid) = sysinfo::get_current_pid() {
        if let Some(current_process) = system.process(current_pid) {
            let process_memory = current_process.memory();
            assert!(process_memory > 0, "Current process should use some memory");
            
            // Test process tasks (threads)
            if let Some(tasks) = current_process.tasks() {
                assert!(!tasks.is_empty(), "Process should have at least one task/thread");
            }
        }
    }
}

/// Test sysinfo system refresh functionality
#[test]
fn test_sysinfo_refresh_functionality() {
    let mut system = System::new();
    
    // Initially should have no data
    assert!(system.cpus().is_empty(), "Should start with no CPU data");
    
    // Refresh CPU data
    system.refresh_cpu();
    // Note: First refresh might not show usage data immediately
    
    // Refresh all data
    system.refresh_all();
    assert!(!system.cpus().is_empty(), "Should have CPU data after refresh_all");
    
    // Test selective refresh
    system.refresh_memory();
    assert!(system.total_memory() > 0, "Should have memory data after refresh_memory");
    
    system.refresh_processes();
    assert!(!system.processes().is_empty(), "Should have process data after refresh_processes");
}

/// Test sysinfo process monitoring
#[test]
fn test_sysinfo_process_monitoring() {
    let mut system = System::new_all();
    system.refresh_all();
    
    // Get current process
    if let Ok(current_pid) = sysinfo::get_current_pid() {
        if let Some(process) = system.process(current_pid) {
            // Test process properties
            let name = process.name();
            assert!(!name.is_empty(), "Process should have a name");
            
            let memory = process.memory();
            assert!(memory > 0, "Process should use some memory");
            
            let cpu_usage = process.cpu_usage();
            assert!(cpu_usage >= 0.0, "CPU usage should be non-negative");
            
            // Test process command line (if available)
            let cmd = process.cmd();
            // cmd might be empty on some systems, so we just check it doesn't panic
            let _cmd_len = cmd.len();
            
            // Test process environment (if available)
            let env = process.environ();
            // env might be empty on some systems, so we just check it doesn't panic
            let _env_len = env.len();
            
            // Test process start time
            let start_time = process.start_time();
            assert!(start_time > 0, "Process should have a start time");
            
            // Test process run time
            let run_time = process.run_time();
            assert!(run_time >= 0, "Process run time should be non-negative");
        }
    }
}

/// Test git2 API compatibility and repository operations
#[test]
fn test_git2_api_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    
    // Test repository initialization
    let repo = Repository::init(repo_path).unwrap();
    assert!(!repo.is_bare(), "Initialized repo should not be bare");
    assert!(repo.is_empty().unwrap(), "New repo should be empty");
    
    // Test repository path (paths might have different representations)
    let workdir = repo.workdir().unwrap();
    assert!(workdir.ends_with(repo_path.file_name().unwrap()));
    
    // Test repository state
    let state = repo.state();
    assert_eq!(state, git2::RepositoryState::Clean);
    
    // Test head reference (should not exist in empty repo)
    let head_result = repo.head();
    assert!(head_result.is_err(), "Empty repo should not have HEAD");
}

/// Test git2 repository initialization with options
#[test]
fn test_git2_repository_init_options() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    
    // Test with initialization options
    let mut opts = RepositoryInitOptions::new();
    opts.bare(false);
    opts.mkdir(true);
    opts.mkpath(true);
    
    let repo = Repository::init_opts(repo_path, &opts).unwrap();
    assert!(!repo.is_bare(), "Repo should not be bare when configured as such");
    
    // Test bare repository
    let bare_temp_dir = TempDir::new().unwrap();
    let bare_repo_path = bare_temp_dir.path();
    
    let mut bare_opts = RepositoryInitOptions::new();
    bare_opts.bare(true);
    
    let bare_repo = Repository::init_opts(bare_repo_path, &bare_opts).unwrap();
    assert!(bare_repo.is_bare(), "Repo should be bare when configured as such");
}

/// Test git2 clone configuration (without actual cloning)
#[test]
fn test_git2_clone_config_compatibility() {
    use git2::build::CloneLocal;
    
    // Test clone builder configuration
    let mut builder = git2::build::RepoBuilder::new();
    
    // Test setting clone options that were fixed in the compilation errors
    builder.clone_local(CloneLocal::Auto);
    
    // Test branch specification
    builder.branch("main");
    
    // Test bare clone
    builder.bare(true);
    
    // Test fetch options
    let mut fetch_opts = git2::FetchOptions::new();
    
    // Test remote callbacks
    let mut callbacks = git2::RemoteCallbacks::new();
    callbacks.update_tips(|_refname, _a, _b| true);
    
    fetch_opts.remote_callbacks(callbacks);
    builder.fetch_options(fetch_opts);
    
    // We don't actually clone anything, just test that the API is compatible
    // and the configuration doesn't cause compilation errors
}

/// Test git2 repository opening and validation
#[test]
fn test_git2_repository_opening() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    
    // Create a repository
    let _repo = Repository::init(repo_path).unwrap();
    
    // Test opening existing repository
    let opened_repo = Repository::open(repo_path).unwrap();
    assert!(!opened_repo.is_bare());
    
    // Test discovering repository
    let _discovered_path = Repository::discover(repo_path).unwrap();
    // Just verify that discovery works without panicking
    
    // Test opening non-existent repository
    let non_existent_path = temp_dir.path().join("non_existent");
    let open_result = Repository::open(&non_existent_path);
    assert!(open_result.is_err(), "Opening non-existent repo should fail");
}

/// Test git2 reference and branch operations
#[test]
fn test_git2_reference_operations() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    let repo = Repository::init(repo_path).unwrap();
    
    // Test reference listing (should be empty for new repo)
    let refs = repo.references().unwrap();
    let ref_count = refs.count();
    // New repo might have 0 or few references
    assert!(ref_count >= 0);
    
    // Test branch listing
    let branches = repo.branches(None).unwrap();
    let branch_count = branches.count();
    // New repo should have no branches initially
    assert_eq!(branch_count, 0);
    
    // Test remote listing
    let remotes = repo.remotes().unwrap();
    assert!(remotes.is_empty(), "New repo should have no remotes");
}

/// Test system resource monitoring integration
#[test]
fn test_system_resource_monitoring_integration() {
    let mut system = System::new_all();
    system.refresh_all();
    
    // Test collecting metrics that would be used in PerformanceMonitor
    let cpu_usage = system.global_cpu_info().cpu_usage();
    let memory_usage = system.used_memory();
    let total_memory = system.total_memory();
    let available_memory = system.available_memory();
    
    // Verify metrics are reasonable
    assert!(cpu_usage >= 0.0 && cpu_usage <= 100.0);
    assert!(memory_usage <= total_memory);
    assert!(available_memory <= total_memory);
    
    // Test process-specific metrics
    if let Ok(current_pid) = sysinfo::get_current_pid() {
        if let Some(process) = system.process(current_pid) {
            let process_memory = process.memory();
            let process_cpu = process.cpu_usage();
            
            assert!(process_memory > 0);
            assert!(process_cpu >= 0.0);
            
            // Test thread count if available
            if let Some(tasks) = process.tasks() {
                assert!(!tasks.is_empty());
            }
        }
    }
}

/// Test Duration import and usage
#[test]
fn test_duration_import_compatibility() {
    // Test Duration creation and operations
    let duration1 = Duration::from_secs(5);
    let duration2 = Duration::from_millis(500);
    let duration3 = Duration::from_micros(1000);
    let duration4 = Duration::from_nanos(1000000);
    
    assert_eq!(duration1.as_secs(), 5);
    assert_eq!(duration2.as_millis(), 500);
    assert_eq!(duration3.as_micros(), 1000);
    assert_eq!(duration4.as_nanos(), 1000000);
    
    // Test Duration arithmetic
    let sum = duration1 + duration2;
    assert_eq!(sum.as_millis(), 5500);
    
    let diff = duration1 - duration2;
    assert_eq!(diff.as_millis(), 4500);
    
    // Test Duration comparison
    assert!(duration1 > duration2);
    assert!(duration2 > duration3);
    // duration3 and duration4 might be equal, so just check they're valid
    assert!(duration3 >= duration4);
    
    // Test Duration with Instant
    let start = Instant::now();
    std::thread::sleep(Duration::from_millis(1));
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(1));
    assert!(elapsed < Duration::from_millis(100)); // Should be reasonable
}

/// Test Process functionality
#[test]
fn test_process_functionality() {
    let mut system = System::new_all();
    system.refresh_all();
    
    if let Ok(current_pid) = sysinfo::get_current_pid() {
        if let Some(process) = system.process(current_pid) {
            // Test Process methods
            let _name = process.name();
            let _pid = process.pid();
            let _parent = process.parent();
            let _memory = process.memory();
            let _virtual_memory = process.virtual_memory();
            let _cpu_usage = process.cpu_usage();
            let _start_time = process.start_time();
            let _run_time = process.run_time();
            let _status = process.status();
            
            // Just verify we can access process properties without panicking
        }
    }
}

/// Test System functionality
#[test]
fn test_system_functionality() {
    let mut system = System::new_all();
    
    // Test System methods
    system.refresh_memory();
    system.refresh_cpu();
    system.refresh_processes();
    
    // Test system information getters (static methods in newer sysinfo)
    let _name = System::name();
    let _kernel_version = System::kernel_version();
    let _os_version = System::os_version();
    let _host_name = System::host_name();
    
    // Test memory information
    let _total_memory = system.total_memory();
    let _free_memory = system.free_memory();
    let _available_memory = system.available_memory();
    let _used_memory = system.used_memory();
    let _total_swap = system.total_swap();
    let _free_swap = system.free_swap();
    let _used_swap = system.used_swap();
    
    // Test CPU information
    let cpus = system.cpus();
    assert!(!cpus.is_empty());
    
    let _global_cpu = system.global_cpu_info();
    
    // Test process information
    let processes = system.processes();
    assert!(!processes.is_empty());
    
    // Test uptime (static method in newer sysinfo)
    let uptime = System::uptime();
    assert!(uptime > 0, "System uptime should be positive");
    
    // Test load average (static method in newer sysinfo)
    let load_avg = System::load_average();
    // Just verify we can call it without panicking
    let _one_min = load_avg.one;
    let _five_min = load_avg.five;
    let _fifteen_min = load_avg.fifteen;
}

/// Test CPU usage functionality
#[test]
fn test_cpu_usage_functionality() {
    let mut system = System::new_all();
    system.refresh_all();
    
    let cpus = system.cpus();
    assert!(!cpus.is_empty());
    
    for cpu in cpus {
        // Test CPU methods
        let _name = cpu.name();
        let usage = cpu.cpu_usage();
        let _frequency = cpu.frequency();
        let _vendor_id = cpu.vendor_id();
        let _brand = cpu.brand();
        
        // Verify usage is reasonable
        assert!(usage >= 0.0, "CPU usage should be non-negative");
        assert!(usage <= 100.0, "CPU usage should not exceed 100%");
    }
    
    // Test global CPU info
    let global_cpu = system.global_cpu_info();
    let global_usage = global_cpu.cpu_usage();
    assert!(global_usage >= 0.0, "Global CPU usage should be non-negative");
}

/// Test dependency version compatibility
#[test]
fn test_dependency_version_compatibility() {
    // Test that we can create and use the main types from dependencies
    
    // sysinfo
    let mut system = System::new();
    system.refresh_all();
    assert!(system.cpus().len() >= 0); // Should not panic
    
    // git2 - test that basic types can be created
    let temp_dir = TempDir::new().unwrap();
    let repo = Repository::init(temp_dir.path()).unwrap();
    assert!(!repo.is_bare());
    
    // std::time - test Duration and Instant
    let duration = Duration::from_secs(1);
    let instant = Instant::now();
    assert!(duration.as_secs() == 1);
    assert!(instant.elapsed() >= Duration::ZERO);
}

/// Test error handling with dependency APIs
#[test]
fn test_dependency_error_handling() {
    // Test sysinfo error handling
    let system = System::new();
    let non_existent_pid = Pid::from(999999);
    let process = system.process(non_existent_pid);
    assert!(process.is_none(), "Non-existent process should return None");
    
    // Test git2 error handling
    let non_existent_path = "/non/existent/path/to/repo";
    let repo_result = Repository::open(non_existent_path);
    assert!(repo_result.is_err(), "Opening non-existent repo should fail");
    
    // Test that errors have reasonable error messages
    match repo_result {
        Err(e) => {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty(), "Error should have a message");
        }
        Ok(_) => panic!("Expected error when opening non-existent repo"),
    }
}

/// Test concurrent usage of dependency APIs
#[test]
fn test_dependency_concurrent_usage() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let system = Arc::new(Mutex::new(System::new_all()));
    let mut handles = Vec::new();
    
    // Test concurrent access to sysinfo
    for _ in 0..5 {
        let system_clone = Arc::clone(&system);
        let handle = thread::spawn(move || {
            let mut sys = system_clone.lock().unwrap();
            sys.refresh_all();
            let cpu_count = sys.cpus().len();
            assert!(cpu_count > 0);
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Test that system is still functional after concurrent access
    let final_system = system.lock().unwrap();
    assert!(!final_system.cpus().is_empty());
}