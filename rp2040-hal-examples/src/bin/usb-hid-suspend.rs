//! A HID keyboard that wakes the host 15 s after the bus suspends, then
//! presses Enter 3 s after the bus resumes. The LED is lit while `UsbDevice`
//! reports `Suspend`.

#![no_std]
#![no_main]

use embedded_hal::digital::OutputPin;
use hal::fugit::MicrosDurationU64;
use panic_halt as _;
use rp2040_hal as hal;
use usb_device::{class_prelude::*, prelude::*};
use usbd_hid::descriptor::generator_prelude::*;
use usbd_hid::descriptor::KeyboardReport;
use usbd_hid::hid_class::HIDClass;

use defmt_rtt as _;

#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_GENERIC_03H;

const XTAL_FREQ_HZ: u32 = 12_000_000;
const WAKE_AFTER: MicrosDurationU64 = MicrosDurationU64::from_secs(15);
/// A host still waking its display drops keys typed straight after the resume.
const TYPE_AFTER: MicrosDurationU64 = MicrosDurationU64::from_secs(3);
const KEY_ENTER: u8 = 0x28;

#[hal::entry]
fn main() -> ! {
    let mut pac = hal::pac::Peripherals::take().unwrap();
    let mut watchdog = hal::Watchdog::new(pac.WATCHDOG);
    let clocks = hal::clocks::init_clocks_and_plls(
        XTAL_FREQ_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .unwrap();
    let timer = hal::Timer::new(pac.TIMER, &mut pac.RESETS, &clocks);
    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );
    let mut led_pin = pins.gpio25.into_push_pull_output();

    let usb_bus = UsbBusAllocator::new(hal::usb::UsbBus::new(
        pac.USBCTRL_REGS,
        pac.USBCTRL_DPRAM,
        clocks.usb_clock,
        true,
        &mut pac.RESETS,
    ));
    let mut keyboard = HIDClass::new_ep_in(&usb_bus, KeyboardReport::desc(), 10);
    let mut usb_dev = UsbDeviceBuilder::new(&usb_bus, UsbVidPid(0x16c0, 0x27dd))
        .strings(&[StringDescriptors::default().product("Remote wakeup")])
        .unwrap()
        .supports_remote_wakeup(true)
        .build();

    let mut suspended_since = None;
    let mut woke_host = false;
    let mut resumed_at = None;
    let mut next_report = None;
    defmt::info!("Booted");

    loop {
        usb_dev.poll(&mut [&mut keyboard]);

        let state = usb_dev.state();
        if state == UsbDeviceState::Suspend {
            led_pin.set_high().unwrap();
        } else {
            led_pin.set_low().unwrap();
        }

        let now = timer.get_counter();
        match state {
            UsbDeviceState::Suspend => {
                if suspended_since.is_none() {
                    defmt::info!("entering suspend at {} !", now);
                };
                let since = *suspended_since.get_or_insert(now);
                if !woke_host && usb_dev.remote_wakeup_enabled() && now - since >= WAKE_AFTER {
                    defmt::info!("Waking host at {}", now);
                    usb_dev.bus().remote_wakeup();
                    woke_host = true;
                }
            }
            UsbDeviceState::Configured => {
                if suspended_since.is_some() {
                    defmt::info!("wake from sleep at {}", now);
                }
                suspended_since = None;
                if woke_host {
                    woke_host = false;
                    defmt::info!("host woken up");
                    resumed_at = Some(now);
                }
                if resumed_at.is_some_and(|at| now - at >= TYPE_AFTER) {
                    defmt::info!(
                        "resumed more than {}secs ago, sending enter!",
                        TYPE_AFTER.as_secs()
                    );
                    resumed_at = None;
                    next_report = Some(KEY_ENTER);
                }
                if let Some(key) = next_report {
                    let mut report = KeyboardReport::default();
                    report.keycodes[0] = key;
                    match keyboard.push_input(&report) {
                        // Send the press, then a release with no keys.
                        Ok(_) => next_report = (key != 0).then_some(0),
                        Err(UsbError::WouldBlock) => {}
                        Err(_) => next_report = None,
                    }
                }
            }
            _ => suspended_since = None,
        }
    }
}
