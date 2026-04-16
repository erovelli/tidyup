# tidyup-storage-sqlite

Default storage backend for [tidyup](https://github.com/erovelli/tidyup). Implements `FileIndex`, `ChangeLog`, and `BackupStore` over a bundled SQLite database with WAL mode and shelf-style backup retention.
