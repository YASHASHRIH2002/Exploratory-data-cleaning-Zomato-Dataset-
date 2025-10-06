class Solution:

    def checkStatus(self, a, b, flag):
        if ((a >= 0 and b < 0) and flag is False):
            return True

        if ((a < 0 and b >= 0) and flag is False):
            return True

        if (a < 0 and b < 0 and flag is True):
            return True

        return False