class Solution:
    def ShortestDistance(self, coordinates):
        shortest_dist = pow((pow((coordinates[2] - coordinates[0]),2) + pow((coordinates[3] - coordinates[1]),2)),0.5)
        print('Distance:', shortest_dist)

    def AcceptInput(self, T):
        for i in range(T):
            coordinates = [0] * 4
            inp = input().split(' ')
            coordinates = list(map(int, inp))
            
            self.ShortestDistance(coordinates)

sol = Solution()
n = int(input('Number of entries : '))
sol.AcceptInput(n)

