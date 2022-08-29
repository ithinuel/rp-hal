//! Timer Peripheral
//!
//! The Timer peripheral on RP2040 consists of a 64-bit counter and 4 alarms.  
//! The Counter is incremented once per microsecond. It obtains its clock source from the watchdog peripheral, you must enable the watchdog before using this peripheral.  
//! Since it would take thousands of years for this counter to overflow you do not need to write logic for dealing with this if using get_counter.  
//!
//! Each of the 4 alarms can match on the lower 32 bits of Counter and trigger an interrupt.
//!
//! See [Chapter 4 Section 6](https://datasheets.raspberrypi.org/rp2040/rp2040_datasheet.pdf) of the datasheet for more details.

use fugit::{Duration, MicrosDurationU64, TimerInstantU64};

use crate::atomic_register_access::{write_bitmask_clear, write_bitmask_set};
use crate::pac::{RESETS, TIMER};
use crate::resets::SubsystemReset;
use core::marker::PhantomData;

fn get_counter(timer: &crate::pac::timer::RegisterBlock) -> TimerInstantU64<1_000_000> {
    let mut hi0 = timer.timerawh.read().bits();
    let timestamp = loop {
        let low = timer.timerawl.read().bits();
        let hi1 = timer.timerawh.read().bits();
        if hi0 == hi1 {
            break (u64::from(hi0) << 32) | u64::from(low);
        }
        hi0 = hi1;
    };
    TimerInstantU64::from_ticks(timestamp)
}
/// Timer peripheral
pub struct Timer {
    timer: TIMER,
    alarms: [bool; 4],
}

impl Timer {
    /// Create a new [`Timer`]
    pub fn new(timer: TIMER, resets: &mut RESETS) -> Self {
        timer.reset_bring_down(resets);
        timer.reset_bring_up(resets);
        Self {
            timer,
            alarms: [true; 4],
        }
    }

    /// Get the current counter value.
    pub fn get_counter(&self) -> TimerInstantU64<1_000_000> {
        get_counter(&self.timer)
    }

    /// Get the value of the least significant word of the counter.
    pub fn get_counter_low(&self) -> u32 {
        self.timer.timerawl.read().bits()
    }

    /// Initialized a Count Down instance without starting it.
    pub fn count_down(&self) -> CountDown<'_> {
        CountDown {
            timer: self,
            period: MicrosDurationU64::nanos(0),
            next_end: None,
        }
    }

    /// Retrieve a reference to alarm 0. Will only return a value the first time this is called
    pub fn alarm_0(&mut self) -> Option<Alarm0> {
        if self.alarms[0] {
            self.alarms[0] = false;
            Some(Alarm0(PhantomData))
        } else {
            None
        }
    }

    /// Retrieve a reference to alarm 1. Will only return a value the first time this is called
    pub fn alarm_1(&mut self) -> Option<Alarm1> {
        if self.alarms[1] {
            self.alarms[1] = false;
            Some(Alarm1(PhantomData))
        } else {
            None
        }
    }

    /// Retrieve a reference to alarm 2. Will only return a value the first time this is called
    pub fn alarm_2(&mut self) -> Option<Alarm2> {
        if self.alarms[2] {
            self.alarms[2] = false;
            Some(Alarm2(PhantomData))
        } else {
            None
        }
    }

    /// Retrieve a reference to alarm 3. Will only return a value the first time this is called
    pub fn alarm_3(&mut self) -> Option<Alarm3> {
        if self.alarms[3] {
            self.alarms[3] = false;
            Some(Alarm3(PhantomData))
        } else {
            None
        }
    }
}

/// Implementation of the embedded_hal::Timer traits using rp2040_hal::timer counter
///
/// ## Usage
/// ```no_run
/// use embedded_hal::timer::{CountDown, Cancel};
/// use fugit::ExtU32;
/// use rp2040_hal;
/// let mut pac = rp2040_hal::pac::Peripherals::take().unwrap();
/// // Configure the Timer peripheral in count-down mode
/// let timer = rp2040_hal::Timer::new(pac.TIMER, &mut pac.RESETS);
/// let mut count_down = timer.count_down();
/// // Create a count_down timer for 500 milliseconds
/// count_down.start(500.millis());
/// // Block until timer has elapsed
/// let _ = nb::block!(count_down.wait());
/// // Restart the count_down timer with a period of 100 milliseconds
/// count_down.start(100.millis());
/// // Cancel it immediately
/// count_down.cancel();
/// ```
pub struct CountDown<'timer> {
    timer: &'timer Timer,
    period: MicrosDurationU64,
    next_end: Option<u64>,
}

impl embedded_hal::timer::CountDown for CountDown<'_> {
    type Time = MicrosDurationU64;

    fn start<T>(&mut self, count: T)
    where
        T: Into<Self::Time>,
    {
        self.period = count.into();
        self.next_end = Some(
            self.timer
                .get_counter()
                .ticks()
                .wrapping_add(self.period.to_micros()),
        );
    }

    fn wait(&mut self) -> nb::Result<(), void::Void> {
        if let Some(end) = self.next_end {
            let ts = self.timer.get_counter().ticks();
            if ts >= end {
                self.next_end = Some(end.wrapping_add(self.period.to_micros()));
                Ok(())
            } else {
                Err(nb::Error::WouldBlock)
            }
        } else {
            panic!("CountDown is not running!");
        }
    }
}

impl embedded_hal::timer::Periodic for CountDown<'_> {}

impl embedded_hal::timer::Cancel for CountDown<'_> {
    type Error = &'static str;

    fn cancel(&mut self) -> Result<(), Self::Error> {
        if self.next_end.is_none() {
            Err("CountDown is not running.")
        } else {
            self.next_end = None;
            Ok(())
        }
    }
}

/// Alarm abstraction.
pub trait Alarm {
    /// Clear the interrupt flag.
    ///
    /// The interrupt is unable to trigger a 2nd time until this interrupt is cleared.
    fn clear_interrupt(&mut self);

    /// Enable this alarm to trigger an interrupt.
    ///
    /// After this interrupt is triggered, make sure to clear the interrupt with [clear_interrupt].
    ///
    /// [clear_interrupt]: #method.clear_interrupt
    fn enable_interrupt(&mut self);

    /// Disable this alarm, preventing it from triggering an interrupt.
    fn disable_interrupt(&mut self);

    /// Schedule the alarm to be finished after `countdown`. If [enable_interrupt] is called,
    /// this will trigger interrupt whenever this time elapses.
    ///
    /// The RP2040 has been observed to take a little while to schedule an alarm. For this
    /// reason, the minimum time that this function accepts is `10.micros()`
    ///
    /// [enable_interrupt]: #method.enable_interrupt
    fn schedule<const NOM: u32, const DENOM: u32>(
        &mut self,
        countdown: Duration<u32, NOM, DENOM>,
    ) -> Result<(), ScheduleAlarmError>;

    /// Schedule the alarm to be finished at `timestamp`. If [enable_interrupt] is called,
    /// this will trigger interrupt whenever the time stamp is reached.
    ///
    /// The RP2040 has been observed to take a little while to schedule an alarm. For this
    /// reason, the minimum time that this function accepts is `10.micros()`
    /// The RP2040 only uses the least significant word of the counter therefore limiting the
    /// alarm to at most `u32::max_value().micros()` in the future.
    ///
    /// [enable_interrupt]: #method.enable_interrupt
    fn schedule_at<const FREQ_HZ: u32>(
        &mut self,
        timestamp: TimerInstantU64<FREQ_HZ>,
    ) -> Result<(), ScheduleAlarmError>;

    /// Return true if this alarm is finished.
    fn finished(&self) -> bool;
}

macro_rules! impl_alarm {
    ($name:ident  { rb: $timer_alarm:ident, int: $int_alarm:ident, int_name: $int_name:tt, armed_bit_mask: $armed_bit_mask: expr }) => {
        /// An alarm that can be used to schedule events in the future. Alarms can also be configured to trigger interrupts.
        pub struct $name(PhantomData<()>);

        impl Alarm for $name {
            /// Clear the interrupt flag. This should be called after interrupt `
            #[doc = $int_name]
            /// ` is called.
            ///
            /// The interrupt is unable to trigger a 2nd time until this interrupt is cleared.
            fn clear_interrupt(&mut self) {
                // safety: TIMER.intr is a write-clear register, so we can atomically clear our interrupt
                // by writing its value to this field
                // Only one instance of this alarm index can exist, and only this alarm interacts with this bit
                // of the TIMER.inte register
                unsafe {
                    let timer = &(*pac::TIMER::ptr());
                    timer.intr.write_with_zero(|w| w.$int_alarm().set_bit());
                }
            }

            /// Enable this alarm to trigger an interrupt. This alarm will trigger `
            #[doc = $int_name]
            /// `.
            ///
            /// After this interrupt is triggered, make sure to clear the interrupt with [clear_interrupt].
            ///
            /// [clear_interrupt]: #method.clear_interrupt
            fn enable_interrupt(&mut self) {
                // safety: using the atomic set alias means we can atomically set our interrupt enable bit.
                // Only one instance of this alarm can exist, and only this alarm interacts with this bit
                // of the TIMER.inte register
                unsafe {
                    let timer = &(*pac::TIMER::ptr());
                    let reg = (&timer.inte).as_ptr();
                    write_bitmask_set(reg, $armed_bit_mask);
                }
            }

            /// Disable this alarm, preventing it from triggering an interrupt.
            fn disable_interrupt(&mut self) {
                // safety: using the atomic set alias means we can atomically clear our interrupt enable bit.
                // Only one instance of this alarm can exist, and only this alarm interacts with this bit
                // of the TIMER.inte register
                unsafe {
                    let timer = &(*pac::TIMER::ptr());
                    let reg = (&timer.inte).as_ptr();
                    write_bitmask_clear(reg, $armed_bit_mask);
                }
            }

            /// Schedule the alarm to be finished after `countdown`. If [enable_interrupt] is called, this will trigger interrupt `
            #[doc = $int_name]
            /// ` whenever this time elapses.
            ///
            /// The RP2040 has been observed to take a little while to schedule an alarm. For this reason, the minimum time that this function accepts is `10.micros()`
            ///
            /// [enable_interrupt]: #method.enable_interrupt
            fn schedule<const NOM: u32, const DENOM: u32>(
                &mut self,
                countdown: Duration<u32, NOM, DENOM>,
            ) -> Result<(), ScheduleAlarmError> {
                let duration = countdown.to_micros();

                const MIN_MICROSECONDS: u32 = 10;
                if duration < MIN_MICROSECONDS {
                    return Err(ScheduleAlarmError::AlarmTooSoon);
                } else {
                    cortex_m::interrupt::free(|_| {
                        // safety: This is a read action and should not have any UB
                        let target_time = unsafe { &*TIMER::ptr() }
                            .timelr
                            .read()
                            .bits()
                            .wrapping_add(duration);

                        // safety: This is the only code in the codebase that accesses memory address $timer_alarm
                        unsafe { &*TIMER::ptr() }
                            .$timer_alarm
                            .write(|w| unsafe { w.bits(target_time) });
                    });
                    Ok(())
                }
            }

            fn schedule_at<const FREQ_HZ: u32>(
                &mut self,
                timestamp: TimerInstantU64<FREQ_HZ>,
            ) -> Result<(), ScheduleAlarmError> {
                const MIN_MICROSECONDS: u64 = 10;
                let timestamp = timestamp.ticks();

                cortex_m::interrupt::free(|_| {
                    // safety: This is a read action and should not have any UB
                    let now = get_counter(unsafe { &*TIMER::ptr() });
                    let duration = timestamp.saturating_sub(now.ticks());

                    if duration > u32::max_value().into() {
                        Err(ScheduleAlarmError::AlarmTooLate)
                    } else if duration < MIN_MICROSECONDS {
                        Err(ScheduleAlarmError::AlarmTooSoon)
                    } else {
                        // safety: This is the only code in the codebase that accesses memory address $timer_alarm
                        unsafe { &*TIMER::ptr() }
                            .$timer_alarm
                            .write(|w| unsafe { w.bits((timestamp & 0xFFFF_FFFF) as u32) });
                        Ok(())
                    }
                })
            }

            /// Return true if this alarm is finished.
            fn finished(&self) -> bool {
                // safety: This is a read action and should not have any UB
                let bits: u32 = unsafe { &*TIMER::ptr() }.armed.read().bits();
                (bits & $armed_bit_mask) == 0
            }
        }
    };
}

/// Errors that can be returned from any of the `AlarmX::schedule` methods.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleAlarmError {
    /// Alarm time is too low. Should be at least 10 microseconds.
    AlarmTooSoon,
    /// Alarm time is too high. Should not be more than `u32::max_value()` in the future.
    AlarmTooLate,
}

impl_alarm!(Alarm0 {
    rb: alarm0,
    int: alarm_0,
    int_name: "TIMER_IRQ_0",
    armed_bit_mask: 0b0001
});

impl_alarm!(Alarm1 {
    rb: alarm1,
    int: alarm_1,
    int_name: "TIMER_IRQ_1",
    armed_bit_mask: 0b0010
});

impl_alarm!(Alarm2 {
    rb: alarm2,
    int: alarm_2,
    int_name: "TIMER_IRQ_2",
    armed_bit_mask: 0b0100
});

impl_alarm!(Alarm3 {
    rb: alarm3,
    int: alarm_3,
    int_name: "TIMER_IRQ_3",
    armed_bit_mask: 0b1000
});
