@dataclass
class GoodBadPts:
    good_pnts: PointPairs
    bad_pnts: PointPairs
    
    def update(self, new_good:PointaArray, new_bad:PointaArray):
        self.good_pnts.previous = self.good_pnts.current
        self.bad_pnts.previous = self.bad_pnts.current
        
        self.good_pnts.current = new_good
        self.bad_pnts.current = new_bad
    
    @staticmethod
    def empty() -> Self:
        empty_log = GoodBadPntLog(
            PointPairs(
                PointaArray(np.array([])),
                PointaArray(np.array([]))
                ),
            PointPairs(
                PointaArray(np.array([])),
                PointaArray(np.array([]))
                )
            )
        return empty_log
        # self._log: GoodBadPntLogs = GoodBadPntLogs.empty()
