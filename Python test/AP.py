'''
This script is code stub for CodeChef problem code APLAM1_PY
Filename:      APLAM1_PY_solution.py
Created:       27/09/2021
Last Modified: 27/09/2021
Author:        e-Yantra Team
'''

# Import reduce module
from functools import reduce

class Solution:
    # Function to generate the A.P. series
    def generate_AP(self, a1, d, n):

        AP_series = []

        for i in range(1, n+1):
            AP_series.append(a1 + (i-1)*d)

        return AP_series
    
    def AP_square(self, AP_series):
        AP_series_square = []
        for i in range(len(AP_series)):
            AP_series_square.append(AP_series[i] ** 2)

        return AP_series_square
    
    def AP_square_sum(self, AP_series_square):
        return sum(AP_series_square)
    
    def AcceptInput(self, T):
        for i in range(T):
            n = input('Enter a1, d, n: ')
            a1, d, n = list(map(int, n.split(' ')))
            
            AP_series = self.generate_AP(a1, d, n)
            for val in AP_series:
                print(val, end=' ')
            print(sep='\n')

            AP_square = self.AP_square(AP_series)
            for val in AP_square:
                print(val, end=' ')
            print(sep='\n')

            AP_sum = self.AP_square_sum(AP_square)
            print(AP_sum)


sol = Solution()
T = int(input())

sol.AcceptInput(T)