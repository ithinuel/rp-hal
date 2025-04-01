//! Shared code and types for Raspberry Pi Silicon UARTS

pub mod common_configs;
mod utils;

pub use utils::*;

/// Trait to handle both underlying devices
pub trait UartDevice: 'static {
    /// Index of the Uart.
    const ID: usize;

    fn as_block(&self) -> &rp_pac_common::uart::RegisterBlock;

    /// The DREQ number for which TX DMA requests are triggered.
    fn tx_dreq() -> u8
    where
        Self: Sized;
    /// The DREQ number for which RX DMA requests are triggered.
    fn rx_dreq() -> u8
    where
        Self: Sized;
}
