/// Register `UARTPERIPHID2` reader
pub type R = crate::register_blocks::R<UARTPERIPHID2_SPEC>;
/// Field `DESIGNER1` reader - These bits read back as 0x4
pub type DESIGNER1_R = crate::register_blocks::FieldReader;
/// Field `REVISION` reader - This field depends on the revision of the UART: r1p0 0x0 r1p1 0x1 r1p3 0x2 r1p4 0x2 r1p5 0x3
pub type REVISION_R = crate::register_blocks::FieldReader;
impl R {
    /// Bits 0:3 - These bits read back as 0x4
    #[inline(always)]
    pub fn designer1(&self) -> DESIGNER1_R {
        DESIGNER1_R::new((self.bits & 0x0f) as u8)
    }
    /// Bits 4:7 - This field depends on the revision of the UART: r1p0 0x0 r1p1 0x1 r1p3 0x2 r1p4 0x2 r1p5 0x3
    #[inline(always)]
    pub fn revision(&self) -> REVISION_R {
        REVISION_R::new(((self.bits >> 4) & 0x0f) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTPERIPHID2")
            .field("revision", &self.revision())
            .field("designer1", &self.designer1())
            .finish()
    }
}
/// UARTPeriphID2 Register  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartperiphid2::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTPERIPHID2_SPEC;
impl crate::register_blocks::RegisterSpec for UARTPERIPHID2_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartperiphid2::R`](R) reader structure
impl crate::register_blocks::Readable for UARTPERIPHID2_SPEC {}
/// `reset()` method sets UARTPERIPHID2 to value 0x34
impl crate::register_blocks::Resettable for UARTPERIPHID2_SPEC {
    const RESET_VALUE: u32 = 0x34;
}
