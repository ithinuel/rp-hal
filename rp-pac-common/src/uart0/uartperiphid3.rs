/// Register `UARTPERIPHID3` reader
pub type R = crate::register_blocks::R<UARTPERIPHID3_SPEC>;
/// Field `CONFIGURATION` reader - These bits read back as 0x00
pub type CONFIGURATION_R = crate::register_blocks::FieldReader;
impl R {
    /// Bits 0:7 - These bits read back as 0x00
    #[inline(always)]
    pub fn configuration(&self) -> CONFIGURATION_R {
        CONFIGURATION_R::new((self.bits & 0xff) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTPERIPHID3")
            .field("configuration", &self.configuration())
            .finish()
    }
}
/// UARTPeriphID3 Register  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartperiphid3::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTPERIPHID3_SPEC;
impl crate::register_blocks::RegisterSpec for UARTPERIPHID3_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartperiphid3::R`](R) reader structure
impl crate::register_blocks::Readable for UARTPERIPHID3_SPEC {}
/// `reset()` method sets UARTPERIPHID3 to value 0
impl crate::register_blocks::Resettable for UARTPERIPHID3_SPEC {}
