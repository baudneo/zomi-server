# User Management
>[!IMPORTANT]
> Remember that if user authentication is disabled in the `server.yml` config file, any username:password combo will work.
> Only when user authentication is enabled, will the server check username and password against the user database.

At the moment, user management is handled on the cli using the `mlapi` script. 
The `user` sub-command is used to manage users.

Passwords are stored as bcrypt hashes. The database is using the python `tinydb` library.

>[!WARNING]
> An assumption is made that since the cli is used for user management, security is greater than an exposed API. 
> Therefore, no password verification is required before making changes to any user. If you require tighter controls, 
> implement linux file permissions for the mlapi script and the server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      config file.

>[!CAUTION]
> `--delete` does not require a password! Secure your config file if you have concerns about unauthorized access.

## Default user
The default user is `imoz` with the password `zomi`. When at least 1 user is created, the default user is disabled. 
There is a mechanism like an anti-lockout rule; if all users are deleted, the default user will be re-created and/or re-enabled.

## Commands
>[!NOTE]
> All of these commands require the `mlapi -C /path/to/config/file.yml user` prefix.

### Create a user
```bash
mlapi -C /path/to/config/file.yml user --create --username <username> --password <password>
```

### Verify username and password
```bash
mlapi -C /path/to/config/file.yml user --verify --username <username> --password <password>
```

### Update user password
```bash
mlapi -C /path/to/config/file.yml user --update --username <username> --password <NEW password>
```

### Delete a user
```bash
mlapi -C /path/to/config/file.yml user --delete --username <username>
```

### List users
```bash
mlapi -C /path/to/config/file.yml user --list
```

### Enable a user
```bash
mlapi -C /path/to/config/file.yml user --enable --username <username>
```

### Disable a user
```bash
mlapi -C /path/to/config/file.yml user --disable --username <username>
```

### Check username
```bash
mlapi -C /path/to/config/file.yml user --get --username <username>
```

# API based user management
This is something that will be worked on in the future.