from context.domains import Reader, File, Printer


class Solution(Reader):
    def __init__(self):
        self.file = File(context='./data/')
        # self.reader = Reader()
        # self.printer = Printer()
        self.crime_rate_columns = ['살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율']
        self.crime_columns = ['살인', '강도', '강간', '절도', '폭력']

    def save_police_pos(self, fname):
        # crime_in_seoul.csv
        file = self.file
        file.fname = fname
        self.print(self.csv(file))


    def save_cctv_pos(self, fname):
        # cctv_in_seoul.csv
        file = self.file
        file.fname = fname
        self.print(self.csv(file))

    def save_police_norm(self):
        # googlemap
        pass

    def folium_test(self):
        pass

    def draw_crime_map(self, fname):
        # geo_simple.json
        file = self.file
        file.fname = fname
        self.print(self.json(file))


if __name__ == '__main__':
    Solution().save_police_pos('crime_in_seoul')
    Solution().save_cctv_pos('cctv_in_seoul')
    Solution().draw_crime_map('geo_simple')
