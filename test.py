from multiprocessing import Pool

foo = {1: []}

def f(x):
    return x

def call_f(pool, s, e):
    #pool = Pool(50)
    res = pool.map(f, range(s, e))
    #pool.close()
    #pool.join()
    return res
    #print foo

def main():
    pool = Pool(50)
    res1 = call_f(pool, 0, 100)
    res2 = call_f(pool,101, 300)
    res3 = call_f(pool,201, 500)
    #pool.join()
    print res1
    print res2
    print res3
    print len(res1), len(res2), len(res3)
   

if __name__ == '__main__':
     main()
