$src = "C:\Users\TylerM\University of Missouri\Sidhu, Jagdeep Singh - ril_v5"
$dst = "E:/D2D/DNA_to_Display\dataset\ril_v5"
$failed = @()
foreach ($d in 'images', 'masks') {
  New-Item -ItemType Directory -Force "$dst\$d" | Out-Null
  $todo = Get-ChildItem "$src\$d" -File | Where-Object { -not (Test-Path "$dst\$d\$($_.Name)") }
  $n = 0
  foreach ($f in $todo) {
    $ok = $false
    for ($try = 1; $try -le 3 -and -not $ok; $try++) {
      try {
        Copy-Item $f.FullName "$dst\$d\$($f.Name).part" -ErrorAction Stop
        Move-Item "$dst\$d\$($f.Name).part" "$dst\$d\$($f.Name)" -Force -ErrorAction Stop
        $ok = $true
        # C: is nearly full: hand the local copy back to the cloud right away.
        attrib.exe -P +U $f.FullName | Out-Null
      } catch { Start-Sleep -Seconds (5 * $try) }
    }
    if (-not $ok) { $failed += "$d\$($f.Name)" }
    $n++
    if ($n % 100 -eq 0) { "{0} {1}: {2} / {3}" -f (Get-Date -Format HH:mm:ss), $d, $n, $todo.Count }
  }
  "{0} {1}: done, {2} copied this run" -f (Get-Date -Format HH:mm:ss), $d, $n
}
foreach ($d in 'images', 'masks') { "{0}: {1} of {2} present" -f $d, (Get-ChildItem "$dst\$d" -File | Where-Object { $_.Extension -ne '.part' }).Count, (Get-ChildItem "$src\$d" -File).Count }
"failed: $($failed.Count)"; $failed
