/// Register `UARTPCELLID1` reader
pub type R = crate::register_blocks::R<UARTPCELLID1_SPEC>;
/// Field `UARTPCELLID1` reader - These bits read back as 0xF0
pub type UARTPCELLID1_R = crate::register_blocks::FieldReader;
impl R {
    /// Bits 0:7 - These bits read back as 0xF0
    #[inline(always)]
    pub fn uartpcellid1(&self) -> UARTPCELLID1_R {
        UARTPCELLID1_R::new((self.bits & 0xff) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTPCELLID1")
            .field("uartpcellid1", &self.uartpcellid1())
            .finish()
    }
}
/// UARTPCellID1 Register  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartpcellid1::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTPCELLID1_SPEC;
impl crate::register_blocks::RegisterSpec for UARTPCELLID1_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartpcellid1::R`](R) reader structure
impl crate::register_blocks::Readable for UARTPCELLID1_SPEC {}
/// `reset()` method sets UARTPCELLID1 to value 0xf0
impl crate::register_blocks::Resettable for UARTPCELLID1_SPEC {
    const RESET_VALUE: u32 = 0xf0;
}
