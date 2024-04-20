use modular_bitfield::prelude::*;
use modular_bitfield::BitfieldSpecifier;

#[derive(BitfieldSpecifier, Copy, Clone, Debug, Eq, PartialEq)]
#[bits = 6]
pub enum Exception {
    /// Uncategorized or unknown reason
    Uncategorized,
    /// Trapped WFI or WFE instruction
    WFxTrap,
    /// Trapped AArch32 MCR or MRC access to CP15
    CP15RTTrap,
    /// Trapped AArch32 MCRR or MRRC access to CP15
    CP15RRTTrap,
    /// Trapped AArch32 MCR or MRC access to CP14
    CP14RTTrap,
    /// Trapped AArch32 LDC or STC access to CP14
    CP14DTTrap,
    /// HCPTR-trapped access to SIMD or FP
    AdvSIMDFPAccessTrap,
    // Trapped BXJ instruction not supported in Armv8
    /// Trapped access to SIMD or FP ID register
    FPIDTrap,
    /// Trapped invalid PAC use
    PACTrap,
    /// Trapped MRRC access to CP14 from AArch32
    CP14RRTTrap,
    /// Illegal Execution state
    IllegalState,
    /// SupervisorCall
    SupervisorCall,
    /// HypervisorCall
    HypervisorCall,
    /// Monitor Call or Trapped SMC instruction
    MonitorCall,
    /// Trapped MRS or MSR system register access
    SystemRegisterTrap,
    /// Trapped invalid ERET use
    ERetTrap,
    /// Instruction Abort or Prefetch Abort
    InstructionAbort,
    /// PC alignment fault
    PCAlignment,
    /// DataAbort
    DataAbort,
    /// Data abort at EL1 reported as being from EL2
    NV2DataAbort,
    /// PAC Authentication failure
    PACFail,
    /// SP alignment fault
    SPAlignment,
    /// IEEE trapped FP exception
    FPTrappedException,
    /// SError interrupt
    SError,
    /// (Hardware) Breakpoint
    Breakpoint,
    /// Software Step
    SoftwareStep,
    /// Watchpoint
    Watchpoint,
    /// Watchpoint at EL1 reported as being from EL2
    NV2Watchpoint,
    /// Software Breakpoint Instruction
    SoftwareBreakpoint,
    /// AArch32 Vector Catch
    VectorCatch,
    /// IRQ interrupt
    IRQ,
    /// HCPTR trapped access to SVE
    SVEAccessTrap,
    /// Branch Target Identification
    BranchTarget,
    /// FIQ interrupt
    FIQ,
}

#[derive(Copy, Clone)]
pub struct ExceptionRecord {
    /// Exception class
    exceptype: Exception,
    /// Syndrome record
    syndrome: u32, // u25
    /// Virtual fault address
    vaddress: u64,
    /// Physical fault address for second stage faults is valid
    ipavalid: bool,
    /// Physical fault address for second stage faults is Non-secure or secure
    ns: bool,
    /// Physical fault address for second stage faults
    ipaddress: u64, // u52
}

pub fn exception_syndrome(exceptype: Exception) -> ExceptionRecord {
    ExceptionRecord {
        exceptype,
        syndrome: 0,
        vaddress: 0,
        ipavalid: false,
        ns: false,
        ipaddress: 0,
    }
}
