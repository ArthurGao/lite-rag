# Relay sync tool reference (fictional example)

Relay copies files between a laptop and a storage bucket. This page lists its error codes, configuration keys, exit codes and releases, one section each.

## Error codes

Every error stops the current run, is written to `relay.log` and is shown by `relay status`.

### E4010

E4010: the bucket name is empty or contains upper-case letters. Rename the bucket in `relay.toml`. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4011

E4011: the access token has expired; run `relay login` again. Tokens last 30 days. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4012

E4012: the access token is valid but lacks write permission on the bucket. Ask the bucket owner to grant the writer role. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4013

E4013: the bucket does not exist in the configured region. Check `net.region` and the bucket name. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4014

E4014: a file is larger than the 5 GB single-upload limit; enable multipart upload. Multipart upload needs v2.3.0 or later. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4015

E4015: the local path is a symbolic link loop. Remove the link or add it to `sync.exclude`. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4016

E4016: the clock on this machine is more than 15 minutes off; sync the system time. Signed requests are refused when the clock drifts. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

### E4017

E4017: two machines wrote the same file at the same time; the newer copy is kept as `name.conflict`. Merge the two copies by hand. Relay stops the current run and writes the error code to `relay.log`; the next run starts from where this one stopped.

## Configuration keys

Keys are set in `relay.toml` or with `relay config set <key> <value>`.

### sync.interval_s

`sync.interval_s`: seconds between two sync runs, default 300. A change takes effect at the start of the next sync run; `relay config get sync.interval_s` prints the current value.

### sync.retry_max

`sync.retry_max`: how many times a failed upload is retried, default 5. A change takes effect at the start of the next sync run; `relay config get sync.retry_max` prints the current value.

### sync.retry_backoff_ms

`sync.retry_backoff_ms`: the first wait before a retry, doubled after each failure, default 200. A change takes effect at the start of the next sync run; `relay config get sync.retry_backoff_ms` prints the current value.

### sync.max_parallel

`sync.max_parallel`: uploads running at once, default 4. A change takes effect at the start of the next sync run; `relay config get sync.max_parallel` prints the current value.

### sync.exclude

`sync.exclude`: glob patterns that are never uploaded, default `[".git/**", "*.tmp"]`. A change takes effect at the start of the next sync run; `relay config get sync.exclude` prints the current value.

### net.proxy_url

`net.proxy_url`: an HTTP proxy for every request, unset by default. A change takes effect at the start of the next sync run; `relay config get net.proxy_url` prints the current value.

### net.timeout_s

`net.timeout_s`: how long one request may take before it is cancelled, default 30. A change takes effect at the start of the next sync run; `relay config get net.timeout_s` prints the current value.

## Exit codes

### Exit code 0

Exit code 0: every file is in sync. Scripts can read the exit code to decide whether to alert someone.

### Exit code 2

Exit code 2: the command line was wrong, for example an unknown flag. Scripts can read the exit code to decide whether to alert someone.

### Exit code 3

Exit code 3: some files failed and were left for the next run. Scripts can read the exit code to decide whether to alert someone.

### Exit code 4

Exit code 4: Relay could not reach the storage service at all. Scripts can read the exit code to decide whether to alert someone.

## Releases

### v2.3.0

v2.3.0 added multipart upload for files above 5 GB. Upgrade with `relay self-update`; settings are kept.

### v2.3.1

v2.3.1 fixed a crash when a file name contained an emoji. Upgrade with `relay self-update`; settings are kept.

### v2.4.0

v2.4.0 replaced the `--force` flag with `--overwrite` and dropped support for Windows 8. Upgrade with `relay self-update`; settings are kept.

## Keeping uploads small

Relay only sends the parts of a file that changed since the last run. A file renamed on the laptop is moved inside the bucket instead of being uploaded again.
