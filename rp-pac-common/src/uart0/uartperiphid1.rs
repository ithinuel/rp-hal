/// Register `UARTPERIPHID1` reader
pub type R = crate::register_blocks::R<UARTPERIPHID1_SPEC>;
/// Field `PARTNUMBER1` reader - These bits read back as 0x0
pub type PARTNUMBER1_R = crate::register_blocks::FieldReader;
/// Field `DESIGNER0` reader - These bits read back as 0x1
pub type DESIGNER0_R = crate::register_blocks::FieldReader;
impl R {
    /// Bits 0:3 - These bits read back as 0x0
    #[inline(always)]
    pub fn partnumber1(&self) -> PARTNUMBER1_R {
        PARTNUMBER1_R::new((self.bits & 0x0f) as u8)
    }
    /// Bits 4:7 - These bits read back as 0x1
    #[inline(always)]
    pub fn designer0(&self) -> DESIGNER0_R {
        DESIGNER0_R::new(((self.bits >> 4) & 0x0f) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTPERIPHID1")
            .field("designer0", &self.designer0())
            .field("partnumber1", &self.partnumber1())
            .finish()
    }
}
/// UARTPeriphID1 Register  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartperiphid1::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTPERIPHID1_SPEC;
impl crate::register_blocks::RegisterSpec for UARTPERIPHID1_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartperiphid1::R`](R) reader structure
impl crate::register_blocks::Readable for UARTPERIPHID1_SPEC {}
/// `reset()` method sets UARTPERIPHID1 to value 0x10
impl crate::register_blocks::Resettable for UARTPERIPHID1_SPEC {
    const RESET_VALUE: u32 = 0x10;
}
