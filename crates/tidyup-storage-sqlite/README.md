# tidyup-storage-sqlite

Default storage backend for [tidyup](https://github.com/erovelli/tidyup). Implements `FileIndex`, `ChangeLog`, `BackupStore`, and `RunLog` over a bundled SQLite database with WAL mode, BLAKE3 content-hash dedup, bundle-atomic shelving, and shelf-style backup retention.
