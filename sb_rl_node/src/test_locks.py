import threading
import time

def foo(lock):
    lock.acquire()
    print("Acquired lock once at ", time.time())
    lock.acquire()
    print("Released by bar at ", time.time())
    lock.release()
    return

def bar(lock):
    lock.release()
    return

if __name__ == "__main__":
    lock=threading.Lock()
    foo_thread=threading.Thread(target=foo, args=(lock,))
    foo_thread.start()
    time.sleep(5)
    bar(lock)
