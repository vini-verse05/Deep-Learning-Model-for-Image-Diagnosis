from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

key = get_random_bytes(16)

def encrypt_image(data):

    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    return cipher.nonce + tag + ciphertext


def decrypt_image(enc_data):

    nonce = enc_data[:16]
    tag = enc_data[16:32]
    ciphertext = enc_data[32:]

    cipher = AES.new(key, AES.MODE_EAX, nonce)

    data = cipher.decrypt_and_verify(ciphertext, tag)

    return data