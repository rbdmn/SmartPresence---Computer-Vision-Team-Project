import time

def timerawal():
    return time.perf_counter()

def timerstoptampilkantimer(start):
    elapsed = time.perf_counter() - start
    formatted = f"{elapsed*1000:.2f} ms"
    print("Elapsed:", formatted)
    return formatted