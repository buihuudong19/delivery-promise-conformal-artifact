# detect_workers.ps1 -- called by run_pipeline.bat at the start of a run.
# Chooses the number of parallel processes of the stages rq1_fit and sens_fit for this machine and prints
#   <workers> <logical processors> <free RAM in GB> <total RAM in GB>
# Rules: each process uses n_jobs = 2 threads (config.yaml) and up to about 1.5 GB of RAM (measured peak 0.8 GB at
# the largest origin, with margin for the tuning grid); 2 logical processors and 3 GB of RAM are kept for Windows;
# there are at most 8 blocks of 4 origins (30 origins), so more than 8 processes cannot be used.
param([double]$GbPerWorker = 1.5, [double]$ReserveGb = 3, [int]$ThreadsPerWorker = 2, [int]$MaxWorkers = 8)
$ErrorActionPreference = 'Stop'
try {
    $t = [Environment]::ProcessorCount
    $os = Get-CimInstance Win32_OperatingSystem
    $free = $os.FreePhysicalMemory / 1MB
    $total = $os.TotalVisibleMemorySize / 1MB
    $byCpu = [math]::Floor(($t - 2) / $ThreadsPerWorker)
    $byRam = [math]::Floor(($free - $ReserveGb) / $GbPerWorker)
    $w = [math]::Max(1, [math]::Min($MaxWorkers, [math]::Min($byCpu, $byRam)))
    $inv = [Globalization.CultureInfo]::InvariantCulture
    Write-Output ([string]::Format($inv, "{0} {1} {2:0.0} {3:0.0}", $w, $t, $free, $total))
} catch {
    Write-Output "1 ? ? ?"
}
