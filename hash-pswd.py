from passlib.hash import pbkdf2_sha256

# Exemple de hachage de mot de passe
password = "1234567890"
hashed_password = pbkdf2_sha256.hash(password)
print(hashed_password)
