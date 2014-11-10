import sys
import fast_imsearch
from fast_imsearch.vptree import *

if __name__ == '__main__':
    # Test: Simple spelling corrector
    
    try:
        import psyco
        psyco.full()
    except: pass
    
    if len(sys.argv) != 2:
        print 'Please supply the filename of your dictionary, eg /usr/share/dict/words'
        sys.exit(1)
        
    comparison_count = 0
    
    def distance(a,b):
        """ Calculates the Levenshtein distance between a and b.
            (from http://hetland.org/python/) """
        global comparison_count
        comparison_count += 1
        
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n

        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]
        
    print 'Load dictionary'
    words = [ item.strip() for item in open(sys.argv[1],'r') if item.strip() ]
    print '%d words' % len(words)
    
    print
    print 'Construct tree'
    tree = VPTree(words, distance, 100)
    print '%d comparisons' % comparison_count
    comparison_count = 0

    print
    print 'Ready to answer queries'    
    while True:
        print
        query = raw_input('query> ').strip()
        if not query: break
        
        n = 0
        for result in tree.find(query):            
            print '% 6d comparisons later... % 5d %s' % (comparison_count, result[1], result[0])
            comparison_count = 0            
            n += 1
            if n >= 5: break
