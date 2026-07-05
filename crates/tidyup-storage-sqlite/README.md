# tidyup-storage-sqlite

Default storage backend for [tidyup](https://github.com/erovelli/tidyup). Implements `FileIndex`, `ChangeLog`, `BackupStore`, and `RunLog` over a bundled SQLite database with WAL mode, BLAKE3 content hashing, bundle-atomic shelving, and shelf-style backup retention. (Content-addressed *dedup* — classify-once-per-unique-hash — is planned, not yet wired: the `files` table is keyed by path/id and the content hash currently powers the apply-time TOCTOU guard.)
