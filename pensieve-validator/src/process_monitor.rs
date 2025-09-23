use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::interval;

/// Comprehensive system and process monitoring
pub struct ProcessMonitor {
    system: Arc<Mutex<System>>,
    monitoring_interval: Duration,
}

/// Detailed system resource snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64, // Unix timestamp for serialization
    pub process_info: ProcessInfo,
    pub system_info: SystemInfo,
    pub disk_info: DiskInfo,
    pub network_info: NetworkInfo,
    pub thermal_info: ThermalInfo,
}

impl SystemSnapshot {
    pub fn new(
        process_info: ProcessInfo,
        system_info: SystemInfo,
        disk_info: DiskInfo,
        network_info: NetworkInfo,
        thermal_info: ThermalInfo,
    ) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            process_info,
            system_info,
            disk_info,
            network_info,
            thermal_info,
        }
    }
}

/// Process-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub memory_kb: u64,
    pub virtual_memory_kb: u64,
    pub cpu_usage_percent: f32,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub status: String,
    pub start_time: u64,
    pub run_time: Duration,
    pub thread_count: u64,
    pub file_descriptors: u64,
}

/// System-wide information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub total_memory_kb: u64,
    pub available_memory_kb: u64,
    pub used_memory_kb: u64,
    pub memory_usage_percent: f64,
    pub total_swap_kb: u64,
    pub used_swap_kb: u64,
    pub cpu_count: usize,
    pub load_average: LoadAverage,
    pub uptime: Duration,
}

/// System load average information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAverage {
    pub one_minute: f64,
    pub five_minute: f64,
    pub fifteen_minute: f64,
}

/// Disk I/O information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub total_read_bytes: u64,
    pub total_write_bytes: u64,
    pub disk_usage: HashMap<String, DiskUsage>,
}

/// Individual disk usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    pub total_space: u64,
    pub available_space: u64,
    pub used_space: u64,
    pub usage_percent: f64,
    pub mount_point: String,
}

/// Network I/O information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub total_bytes_received: u64,
    pub total_bytes_transmitted: u64,
    pub interfaces: HashMap<String, NetworkInterface>,
}

/// Individual network interface statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub bytes_received: u64,
    pub bytes_transmitted: u64,
    pub packets_received: u64,
    pub packets_transmitted: u64,
    pub errors_received: u64,
    pub errors_transmitted: u64,
}

/// System thermal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    pub components: HashMap<String, ComponentInfo>,
    pub max_temperature: f32,
    pub average_temperature: f32,
}

/// Individual thermal component information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub temperature: f32,
    pub max_temperature: f32,
    pub critical_temperature: Option<f32>,
    pub label: String,
}

/// Aggregated monitoring results over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringResults {
    pub snapshots: Vec<SystemSnapshot>,
    pub summary: MonitoringSummary,
    pub alerts: Vec<MonitoringAlert>,
    pub performance_analysis: PerformanceAnalysis,
}

/// Summary statistics from monitoring session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSummary {
    pub duration: Duration,
    pub snapshot_count: usize,
    pub peak_memory_usage: u64,
    pub average_memory_usage: u64,
    pub peak_cpu_usage: f32,
    pub average_cpu_usage: f32,
    pub total_disk_read: u64,
    pub total_disk_write: u64,
    pub peak_temperature: f32,
    pub memory_efficiency: f64,
    pub cpu_efficiency: f64,
}

/// Monitoring alerts for threshold violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringAlert {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64, // Unix timestamp for serialization
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
}

impl MonitoringAlert {
    pub fn new(
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        value: f64,
        threshold: f64,
    ) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            alert_type,
            severity,
            message,
            value,
            threshold,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AlertType {
    HighMemoryUsage,
    HighCpuUsage,
    HighDiskUsage,
    HighTemperature,
    ProcessUnresponsive,
    DiskSpaceLow,
    SwapUsageHigh,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Performance analysis based on monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub resource_utilization_score: f64, // 0.0 - 1.0
    pub stability_score: f64,            // 0.0 - 1.0 (low variance = high stability)
    pub efficiency_score: f64,           // 0.0 - 1.0
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub recommendations: Vec<String>,
}

/// Identified performance bottlenecks
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub resource_type: String,
    pub severity: f64, // 0.0 - 1.0
    pub description: String,
    pub duration: Duration,
    pub impact_assessment: String,
}

/// Configuration for process monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub interval_ms: u64,
    pub memory_threshold_percent: f64,
    pub cpu_threshold_percent: f64,
    pub disk_threshold_percent: f64,
    pub temperature_threshold_celsius: f32,
    pub enable_detailed_disk_monitoring: bool,
    pub enable_network_monitoring: bool,
    pub enable_thermal_monitoring: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            interval_ms: 500,
            memory_threshold_percent: 80.0,
            cpu_threshold_percent: 90.0,
            disk_threshold_percent: 85.0,
            temperature_threshold_celsius: 80.0,
            enable_detailed_disk_monitoring: true,
            enable_network_monitoring: true,
            enable_thermal_monitoring: true,
        }
    }
}

impl ProcessMonitor {
    /// Create a new ProcessMonitor with default configuration
    pub fn new() -> Self {
        Self::with_config(MonitoringConfig::default())
    }

    /// Create a new ProcessMonitor with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            system: Arc::new(Mutex::new(System::new_all())),
            monitoring_interval: Duration::from_millis(config.interval_ms),
        }
    }

    /// Start monitoring a specific process
    pub async fn monitor_process(
        &self,
        pid: u32,
        config: MonitoringConfig,
    ) -> std::result::Result<(mpsc::Receiver<SystemSnapshot>, JoinHandle<()>), Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = mpsc::channel(1000);
        let system = Arc::clone(&self.system);
        let interval_duration = Duration::from_millis(config.interval_ms);
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            let process_pid = Pid::from(pid as usize);
            let start_time = Instant::now();
            
            loop {
                interval.tick().await;
                
                // Refresh system information
                {
                    let mut sys = system.lock().unwrap();
                    sys.refresh_all();
                }
                
                // Collect system snapshot
                let snapshot = {
                    let sys = system.lock().unwrap();
                    Self::collect_system_snapshot(&sys, process_pid, start_time, &config)
                };
                
                match snapshot {
                    Some(snapshot) => {
                        if tx.send(snapshot).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    None => {
                        // Process no longer exists
                        break;
                    }
                }
            }
        });
        
        Ok((rx, handle))
    }

    /// Collect a comprehensive system snapshot
    fn collect_system_snapshot(
        system: &System,
        pid: Pid,
        start_time: Instant,
        config: &MonitoringConfig,
    ) -> Option<SystemSnapshot> {
        let process = system.process(pid)?;
        
        // Process information
        let process_info = ProcessInfo {
            pid: pid.as_u32() as u32,
            memory_kb: process.memory(),
            virtual_memory_kb: process.virtual_memory(),
            cpu_usage_percent: process.cpu_usage(),
            disk_read_bytes: process.disk_usage().read_bytes,
            disk_write_bytes: process.disk_usage().written_bytes,
            status: format!("{:?}", process.status()),
            start_time: process.start_time(),
            run_time: start_time.elapsed(),
            thread_count: 1, // sysinfo doesn't provide thread count directly
            file_descriptors: 0, // Would need platform-specific implementation
        };

        // System information
        let total_memory = system.total_memory();
        let available_memory = system.available_memory();
        let used_memory = system.used_memory();
        let memory_usage_percent = if total_memory > 0 {
            (used_memory as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        let system_info = SystemInfo {
            total_memory_kb: total_memory,
            available_memory_kb: available_memory,
            used_memory_kb: used_memory,
            memory_usage_percent,
            total_swap_kb: system.total_swap(),
            used_swap_kb: system.used_swap(),
            cpu_count: system.cpus().len(),
            load_average: LoadAverage {
                one_minute: System::load_average().one,
                five_minute: System::load_average().five,
                fifteen_minute: System::load_average().fifteen,
            },
            uptime: Duration::from_secs(System::uptime()),
        };

        // Disk information (if enabled)
        let disk_info = if config.enable_detailed_disk_monitoring {
            Self::collect_disk_info(system)
        } else {
            DiskInfo {
                total_read_bytes: 0,
                total_write_bytes: 0,
                disk_usage: HashMap::new(),
            }
        };

        // Network information (if enabled)
        let network_info = if config.enable_network_monitoring {
            Self::collect_network_info(system)
        } else {
            NetworkInfo {
                total_bytes_received: 0,
                total_bytes_transmitted: 0,
                interfaces: HashMap::new(),
            }
        };

        // Thermal information (if enabled)
        let thermal_info = if config.enable_thermal_monitoring {
            Self::collect_thermal_info(system)
        } else {
            ThermalInfo {
                components: HashMap::new(),
                max_temperature: 0.0,
                average_temperature: 0.0,
            }
        };

        Some(SystemSnapshot::new(
            process_info,
            system_info,
            disk_info,
            network_info,
            thermal_info,
        ))
    }

    /// Collect disk usage information
    fn collect_disk_info(_system: &System) -> DiskInfo {
        let disk_usage = HashMap::new();
        // Note: sysinfo API has changed, disk monitoring would need platform-specific implementation
        // For now, return empty data structure
        
        DiskInfo {
            total_read_bytes: 0,
            total_write_bytes: 0,
            disk_usage,
        }
    }

    /// Collect network interface information
    fn collect_network_info(_system: &System) -> NetworkInfo {
        let interfaces = HashMap::new();
        // Note: sysinfo API has changed, network monitoring would need platform-specific implementation
        // For now, return empty data structure
        
        NetworkInfo {
            total_bytes_received: 0,
            total_bytes_transmitted: 0,
            interfaces,
        }
    }

    /// Collect thermal/temperature information
    fn collect_thermal_info(_system: &System) -> ThermalInfo {
        let components = HashMap::new();
        // Note: sysinfo API has changed, thermal monitoring would need platform-specific implementation
        // For now, return empty data structure
        
        ThermalInfo {
            components,
            max_temperature: 0.0,
            average_temperature: 0.0,
        }
    }

    /// Analyze monitoring results and generate comprehensive report
    pub fn analyze_monitoring_results(
        &self,
        snapshots: Vec<SystemSnapshot>,
        config: &MonitoringConfig,
    ) -> MonitoringResults {
        if snapshots.is_empty() {
            return MonitoringResults {
                snapshots: Vec::new(),
                summary: MonitoringSummary {
                    duration: Duration::from_secs(0),
                    snapshot_count: 0,
                    peak_memory_usage: 0,
                    average_memory_usage: 0,
                    peak_cpu_usage: 0.0,
                    average_cpu_usage: 0.0,
                    total_disk_read: 0,
                    total_disk_write: 0,
                    peak_temperature: 0.0,
                    memory_efficiency: 0.0,
                    cpu_efficiency: 0.0,
                },
                alerts: Vec::new(),
                performance_analysis: PerformanceAnalysis {
                    resource_utilization_score: 0.0,
                    stability_score: 0.0,
                    efficiency_score: 0.0,
                    bottlenecks: Vec::new(),
                    recommendations: Vec::new(),
                },
            };
        }

        let summary = self.calculate_summary(&snapshots);
        let alerts = self.generate_alerts(&snapshots, config);
        let performance_analysis = self.analyze_performance(&snapshots);

        MonitoringResults {
            snapshots,
            summary,
            alerts,
            performance_analysis,
        }
    }

    /// Calculate summary statistics
    fn calculate_summary(&self, snapshots: &[SystemSnapshot]) -> MonitoringSummary {
        let duration = if snapshots.len() > 1 {
            snapshots.last().unwrap().timestamp.duration_since(snapshots.first().unwrap().timestamp)
        } else {
            Duration::from_secs(0)
        };

        let memory_values: Vec<u64> = snapshots.iter().map(|s| s.process_info.memory_kb).collect();
        let cpu_values: Vec<f32> = snapshots.iter().map(|s| s.process_info.cpu_usage_percent).collect();
        let temp_values: Vec<f32> = snapshots.iter().map(|s| s.thermal_info.max_temperature).collect();

        let peak_memory = memory_values.iter().max().copied().unwrap_or(0);
        let average_memory = if !memory_values.is_empty() {
            memory_values.iter().sum::<u64>() / memory_values.len() as u64
        } else {
            0
        };

        let peak_cpu = cpu_values.iter().fold(0.0f32, |a, &b| a.max(b));
        let average_cpu = if !cpu_values.is_empty() {
            cpu_values.iter().sum::<f32>() / cpu_values.len() as f32
        } else {
            0.0
        };

        let peak_temperature = temp_values.iter().fold(0.0f32, |a, &b| a.max(b));

        // Calculate efficiency scores
        let memory_efficiency = self.calculate_memory_efficiency(&memory_values);
        let cpu_efficiency = self.calculate_cpu_efficiency(&cpu_values);

        MonitoringSummary {
            duration,
            snapshot_count: snapshots.len(),
            peak_memory_usage: peak_memory,
            average_memory_usage: average_memory,
            peak_cpu_usage: peak_cpu,
            average_cpu_usage: average_cpu,
            total_disk_read: snapshots.last().map(|s| s.process_info.disk_read_bytes).unwrap_or(0),
            total_disk_write: snapshots.last().map(|s| s.process_info.disk_write_bytes).unwrap_or(0),
            peak_temperature,
            memory_efficiency,
            cpu_efficiency,
        }
    }

    /// Generate alerts based on threshold violations
    fn generate_alerts(&self, snapshots: &[SystemSnapshot], config: &MonitoringConfig) -> Vec<MonitoringAlert> {
        let mut alerts = Vec::new();

        for snapshot in snapshots {
            // Memory usage alerts
            if snapshot.system_info.memory_usage_percent > config.memory_threshold_percent {
                alerts.push(MonitoringAlert::new(
                    AlertType::HighMemoryUsage,
                    if snapshot.system_info.memory_usage_percent > 95.0 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    format!("High system memory usage: {:.1}%", snapshot.system_info.memory_usage_percent),
                    snapshot.system_info.memory_usage_percent,
                    config.memory_threshold_percent,
                ));
            }

            // CPU usage alerts
            if snapshot.process_info.cpu_usage_percent > config.cpu_threshold_percent as f32 {
                alerts.push(MonitoringAlert::new(
                    AlertType::HighCpuUsage,
                    AlertSeverity::Warning,
                    format!("High CPU usage: {:.1}%", snapshot.process_info.cpu_usage_percent),
                    snapshot.process_info.cpu_usage_percent as f64,
                    config.cpu_threshold_percent,
                ));
            }

            // Temperature alerts
            if snapshot.thermal_info.max_temperature > config.temperature_threshold_celsius {
                alerts.push(MonitoringAlert::new(
                    AlertType::HighTemperature,
                    if snapshot.thermal_info.max_temperature > 90.0 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    format!("High temperature: {:.1}°C", snapshot.thermal_info.max_temperature),
                    snapshot.thermal_info.max_temperature as f64,
                    config.temperature_threshold_celsius as f64,
                ));
            }

            // Disk usage alerts
            for (mount_point, disk_usage) in &snapshot.disk_info.disk_usage {
                if disk_usage.usage_percent > config.disk_threshold_percent {
                    alerts.push(MonitoringAlert::new(
                        AlertType::HighDiskUsage,
                        if disk_usage.usage_percent > 95.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        format!("High disk usage on {}: {:.1}%", mount_point, disk_usage.usage_percent),
                        disk_usage.usage_percent,
                        config.disk_threshold_percent,
                    ));
                }
            }
        }

        alerts
    }

    /// Analyze performance patterns and identify bottlenecks
    fn analyze_performance(&self, snapshots: &[SystemSnapshot]) -> PerformanceAnalysis {
        let memory_values: Vec<u64> = snapshots.iter().map(|s| s.process_info.memory_kb).collect();
        let cpu_values: Vec<f32> = snapshots.iter().map(|s| s.process_info.cpu_usage_percent).collect();

        // Calculate resource utilization score
        let avg_memory_percent = if !snapshots.is_empty() {
            let total_memory = snapshots[0].system_info.total_memory_kb as f64;
            let avg_memory = memory_values.iter().sum::<u64>() as f64 / memory_values.len() as f64;
            (avg_memory / total_memory) * 100.0
        } else {
            0.0
        };

        let avg_cpu_percent = if !cpu_values.is_empty() {
            cpu_values.iter().sum::<f32>() as f64 / cpu_values.len() as f64
        } else {
            0.0
        };

        let resource_utilization_score = ((avg_memory_percent + avg_cpu_percent) / 200.0).min(1.0);

        // Calculate stability score (lower variance = higher stability)
        let stability_score = self.calculate_stability_score(&memory_values, &cpu_values);

        // Calculate efficiency score
        let efficiency_score = (self.calculate_memory_efficiency(&memory_values) + 
                               self.calculate_cpu_efficiency(&cpu_values)) / 2.0;

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(snapshots);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&bottlenecks, avg_memory_percent, avg_cpu_percent);

        PerformanceAnalysis {
            resource_utilization_score,
            stability_score,
            efficiency_score,
            bottlenecks,
            recommendations,
        }
    }

    /// Calculate memory efficiency score
    fn calculate_memory_efficiency(&self, memory_values: &[u64]) -> f64 {
        if memory_values.is_empty() {
            return 1.0;
        }

        // Efficiency is inversely related to memory usage variance
        // Lower variance = more efficient memory usage
        let mean = memory_values.iter().sum::<u64>() as f64 / memory_values.len() as f64;
        let variance = memory_values.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / memory_values.len() as f64;
        
        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Calculate CPU efficiency score
    fn calculate_cpu_efficiency(&self, cpu_values: &[f32]) -> f64 {
        if cpu_values.is_empty() {
            return 1.0;
        }

        // Efficiency is based on consistent CPU usage without spikes
        let mean = cpu_values.iter().sum::<f32>() as f64 / cpu_values.len() as f64;
        let variance = cpu_values.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / cpu_values.len() as f64;
        
        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Calculate overall stability score
    fn calculate_stability_score(&self, memory_values: &[u64], cpu_values: &[f32]) -> f64 {
        let memory_stability = self.calculate_memory_efficiency(memory_values);
        let cpu_stability = self.calculate_cpu_efficiency(cpu_values);
        (memory_stability + cpu_stability) / 2.0
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, snapshots: &[SystemSnapshot]) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        if snapshots.is_empty() {
            return bottlenecks;
        }

        // Analyze memory bottlenecks
        let high_memory_count = snapshots.iter()
            .filter(|s| s.system_info.memory_usage_percent > 80.0)
            .count();
        
        if high_memory_count > snapshots.len() / 2 {
            bottlenecks.push(PerformanceBottleneck {
                resource_type: "Memory".to_string(),
                severity: (high_memory_count as f64 / snapshots.len() as f64),
                description: "Sustained high memory usage detected".to_string(),
                duration: Duration::from_secs((snapshots.len() as u64 * 500) / 1000), // Approximate
                impact_assessment: "May cause system slowdown and swapping".to_string(),
            });
        }

        // Analyze CPU bottlenecks
        let high_cpu_count = snapshots.iter()
            .filter(|s| s.process_info.cpu_usage_percent > 80.0)
            .count();
        
        if high_cpu_count > snapshots.len() / 4 {
            bottlenecks.push(PerformanceBottleneck {
                resource_type: "CPU".to_string(),
                severity: (high_cpu_count as f64 / snapshots.len() as f64),
                description: "High CPU usage periods detected".to_string(),
                duration: Duration::from_secs((high_cpu_count as u64 * 500) / 1000),
                impact_assessment: "May cause processing delays and system responsiveness issues".to_string(),
            });
        }

        // Analyze disk bottlenecks
        for snapshot in snapshots {
            for (mount_point, disk_usage) in &snapshot.disk_info.disk_usage {
                if disk_usage.usage_percent > 90.0 {
                    bottlenecks.push(PerformanceBottleneck {
                        resource_type: format!("Disk ({})", mount_point),
                        severity: disk_usage.usage_percent / 100.0,
                        description: format!("Very high disk usage on {}", mount_point),
                        duration: Duration::from_secs(1), // Single snapshot
                        impact_assessment: "May cause I/O delays and application slowdown".to_string(),
                    });
                    break; // Only report once per mount point
                }
            }
        }

        bottlenecks
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self, bottlenecks: &[PerformanceBottleneck], avg_memory: f64, avg_cpu: f64) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory recommendations
        if avg_memory > 80.0 {
            recommendations.push("Consider increasing system memory or optimizing memory usage".to_string());
        }
        if avg_memory > 95.0 {
            recommendations.push("Critical: System is running out of memory. Immediate action required".to_string());
        }

        // CPU recommendations
        if avg_cpu > 80.0 {
            recommendations.push("Consider optimizing CPU-intensive operations or upgrading CPU".to_string());
        }

        // Bottleneck-specific recommendations
        for bottleneck in bottlenecks {
            match bottleneck.resource_type.as_str() {
                "Memory" => {
                    recommendations.push("Implement memory pooling or reduce memory allocations".to_string());
                }
                "CPU" => {
                    recommendations.push("Consider parallel processing or algorithm optimization".to_string());
                }
                resource if resource.starts_with("Disk") => {
                    recommendations.push(format!("Free up space on {} or move data to another volume", resource));
                }
                _ => {}
            }
        }

        if recommendations.is_empty() {
            recommendations.push("System performance is within acceptable parameters".to_string());
        }

        recommendations
    }
}

impl Default for ProcessMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_monitor_creation() {
        let monitor = ProcessMonitor::new();
        assert!(monitor.monitoring_interval.as_millis() > 0);
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert_eq!(config.interval_ms, 500);
        assert_eq!(config.memory_threshold_percent, 80.0);
        assert_eq!(config.cpu_threshold_percent, 90.0);
        assert!(config.enable_detailed_disk_monitoring);
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        let monitor = ProcessMonitor::new();
        
        // Test with consistent memory usage (high efficiency)
        let consistent_values = vec![100, 101, 99, 100, 102];
        let efficiency = monitor.calculate_memory_efficiency(&consistent_values);
        assert!(efficiency > 0.9); // Should be high efficiency
        
        // Test with highly variable memory usage (low efficiency)
        let variable_values = vec![100, 500, 50, 800, 25];
        let efficiency = monitor.calculate_memory_efficiency(&variable_values);
        assert!(efficiency < 0.5); // Should be low efficiency
    }

    #[test]
    fn test_alert_generation() {
        let monitor = ProcessMonitor::new();
        let config = MonitoringConfig {
            memory_threshold_percent: 80.0,
            cpu_threshold_percent: 90.0,
            ..Default::default()
        };

        let snapshot = SystemSnapshot::new(
            ProcessInfo {
                pid: 1234,
                memory_kb: 1000000,
                virtual_memory_kb: 2000000,
                cpu_usage_percent: 95.0, // Above threshold
                disk_read_bytes: 0,
                disk_write_bytes: 0,
                status: "Running".to_string(),
                start_time: 0,
                run_time: Duration::from_secs(60),
                thread_count: 1,
                file_descriptors: 10,
            },
            SystemInfo {
                total_memory_kb: 8000000,
                available_memory_kb: 1000000,
                used_memory_kb: 7000000,
                memory_usage_percent: 87.5, // Above threshold
                total_swap_kb: 2000000,
                used_swap_kb: 100000,
                cpu_count: 4,
                load_average: LoadAverage {
                    one_minute: 1.0,
                    five_minute: 1.2,
                    fifteen_minute: 1.1,
                },
                uptime: Duration::from_secs(3600),
            },
            DiskInfo {
                total_read_bytes: 0,
                total_write_bytes: 0,
                disk_usage: HashMap::new(),
            },
            NetworkInfo {
                total_bytes_received: 0,
                total_bytes_transmitted: 0,
                interfaces: HashMap::new(),
            },
            ThermalInfo {
                components: HashMap::new(),
                max_temperature: 0.0,
                average_temperature: 0.0,
            },
        );

        let alerts = monitor.generate_alerts(&[snapshot], &config);
        
        // Should generate alerts for both high memory and high CPU usage
        assert!(alerts.len() >= 2);
        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::HighMemoryUsage)));
        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::HighCpuUsage)));
    }
}