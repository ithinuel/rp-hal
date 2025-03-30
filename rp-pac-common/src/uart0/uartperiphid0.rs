/// Register `UARTPERIPHID0` reader
pub type R = crate::register_blocks::R<UARTPERIPHID0_SPEC>;
/// Field `PARTNUMBER0` reader - These bits read back as 0x11
pub type PARTNUMBER0_R = crate::register_blocks::FieldReader;
impl R {
    /// Bits 0:7 - These bits read back as 0x11
    #[inline(always)]
    pub fn partnumber0(&self) -> PARTNUMBER0_R {
        PARTNUMBER0_R::new((self.bits & 0xff) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTPERIPHID0")
            .field("partnumber0", &self.partnumber0())
            .finish()
    }
}
/// UARTPeriphID0 Register  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartperiphid0::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTPERIPHID0_SPEC;
impl crate::register_blocks::RegisterSpec for UARTPERIPHID0_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartperiphid0::R`](R) reader structure
impl crate::register_blocks::Readable for UARTPERIPHID0_SPEC {}
/// `reset()` method sets UARTPERIPHID0 to value 0x11
impl crate::register_blocks::Resettable for UARTPERIPHID0_SPEC {
    const RESET_VALUE: u32 = 0x11;
}
