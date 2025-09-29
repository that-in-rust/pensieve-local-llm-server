//! Timestamp conversion utilities for serialization compatibility
//!
//! This module provides utilities for converting between `Instant` and `SystemTime`
//! to enable serialization of time-based data structures.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Convert `Instant` to `SystemTime` for serialization
/// 
/// Note: This conversion is approximate since `Instant` is relative to an arbitrary point
/// and `SystemTime` is relative to the Unix epoch. The conversion uses the current time
/// as a reference point.
pub fn instant_to_system_time(instant: Instant) -> SystemTime {
    let now_instant = Instant::now();
    let now_system = SystemTime::now();
    
    if instant <= now_instant {
        let duration_ago = now_instant.duration_since(instant);
        now_system.checked_sub(duration_ago).unwrap_or(now_system)
    } else {
        let duration_ahead = instant.duration_since(now_instant);
        now_system.checked_add(duration_ahead).unwrap_or(now_system)
    }
}

/// Convert `SystemTime` to `Instant` for internal use
/// 
/// Note: This conversion is approximate and should only be used when necessary.
/// The conversion uses the current time as a reference point.
pub fn system_time_to_instant(system_time: SystemTime) -> Instant {
    let now_system = SystemTime::now();
    let now_instant = Instant::now();
    
    match system_time.duration_since(now_system) {
        Ok(duration_ahead) => now_instant + duration_ahead,
        Err(e) => {
            let duration_ago = e.duration();
            now_instant.checked_sub(duration_ago).unwrap_or(now_instant)
        }
    }
}

/// Serde serialization module for `SystemTime` as Unix timestamp
pub mod system_time_serde {
    use super::*;
    
    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Serde serialization module for `Option<SystemTime>` as Unix timestamp
pub mod optional_system_time_serde {
    use super::*;
    
    pub fn serialize<S>(time: &Option<SystemTime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match time {
            Some(t) => Some(t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()).serialize(serializer),
            None => None::<u64>.serialize(serializer),
        }
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<SystemTime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs_opt = Option::<u64>::deserialize(deserializer)?;
        Ok(secs_opt.map(|secs| UNIX_EPOCH + Duration::from_secs(secs)))
    }
}

/// Serde serialization module for `Duration` as milliseconds
pub mod duration_millis_serde {
    use super::*;
    
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u128::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis as u64))
    }
}

/// Serde serialization module for `Duration` as seconds (f64)
pub mod duration_secs_serde {
    use super::*;
    
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    
    #[derive(Serialize, Deserialize)]
    struct TestStruct {
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
        #[serde(with = "optional_system_time_serde")]
        optional_timestamp: Option<SystemTime>,
        #[serde(with = "duration_millis_serde")]
        duration_millis: Duration,
        #[serde(with = "duration_secs_serde")]
        duration_secs: Duration,
    }
    
    #[test]
    fn test_system_time_serialization() {
        let test_data = TestStruct {
            timestamp: SystemTime::now(),
            optional_timestamp: Some(SystemTime::now()),
            duration_millis: Duration::from_millis(1500),
            duration_secs: Duration::from_secs_f64(2.5),
        };
        
        let json = serde_json::to_string(&test_data).unwrap();
        let deserialized: TestStruct = serde_json::from_str(&json).unwrap();
        
        // Check that serialization/deserialization works
        assert!(deserialized.timestamp.duration_since(UNIX_EPOCH).is_ok());
        assert!(deserialized.optional_timestamp.is_some());
        assert_eq!(deserialized.duration_millis, Duration::from_millis(1500));
        assert_eq!(deserialized.duration_secs, Duration::from_secs_f64(2.5));
    }
    
    #[test]
    fn test_instant_system_time_conversion() {
        let instant = Instant::now();
        let system_time = instant_to_system_time(instant);
        let back_to_instant = system_time_to_instant(system_time);
        
        // The conversion should be approximately correct (within a reasonable margin)
        let diff = if back_to_instant >= instant {
            back_to_instant.duration_since(instant)
        } else {
            instant.duration_since(back_to_instant)
        };
        
        // Should be very close (within 1ms due to timing differences)
        assert!(diff < Duration::from_millis(10));
    }
    
    #[test]
    fn test_optional_system_time_none() {
        #[derive(Serialize, Deserialize)]
        struct TestOptional {
            #[serde(with = "optional_system_time_serde")]
            timestamp: Option<SystemTime>,
        }
        
        let test_data = TestOptional { timestamp: None };
        let json = serde_json::to_string(&test_data).unwrap();
        let deserialized: TestOptional = serde_json::from_str(&json).unwrap();
        
        assert!(deserialized.timestamp.is_none());
    }
}