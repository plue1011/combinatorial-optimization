import numpy as np
import matplotlib.pyplot as plt
import pulp

def compute_distance(lon_a, lat_a, lon_b, lat_b):
    """
    緯度経度から距離を計算する
    地点A(経度lon_a, 緯度lat_a)、地点B(経度lon_b, 緯度lat_b)
    
    Parameters
    ----------
    lon_a : float
        地点1の経度
    lat_a : float
        地点1の緯度
    lon_b : float
        地点2の経度
    lat_b : float
        地点2の緯度
    
    Returns
    -------
    rho : float
        ２地点間の距離(km)
    """
    if (lon_a == lon_b) and (lat_a == lat_b):
        return 0.0
    
    ra = 6378.140  # equatorial radius (km)
    rb = 6356.755  # polar radius (km)
    F = (ra - rb) / ra # flattening of the earth
    rad_lat_a = np.radians(lat_a)
    rad_lon_a = np.radians(lon_a)
    rad_lat_b = np.radians(lat_b)
    rad_lon_b = np.radians(lon_b)
    pa = np.arctan(rb / ra * np.tan(rad_lat_a))
    pb = np.arctan(rb / ra * np.tan(rad_lat_b))
    xx = np.arccos(np.sin(pa) * np.sin(pb) + np.cos(pa) * np.cos(pb) * np.cos(rad_lon_a - rad_lon_b))
    c1 = (np.sin(xx) - xx) * (np.sin(pa) + np.sin(pb))**2 / np.cos(xx / 2)**2
    c2 = (np.sin(xx) + xx) * (np.sin(pa) - np.sin(pb))**2 / np.sin(xx / 2)**2
    dr = F / 8 * (c1 - c2)
    rho = ra * (xx + dr)
    return rho

class PCTSP:
    """
    prize collecting traveling salesman problem

    Attributes
    ----------
    I : array of int
        観光地名id(0以上の連続した数列)
    a : array of int or float
        観光地の満足度
    b : array of int or float
        観光地の滞在時間
    pos : ndarray of float
        [[x座標, y座標]_pos1, [x座標, y座標]_pos2]
        観光地の座標
    speed : float or int
        移動速度
    c : array of int
        観光地ごとの費用
    m : ndarray(N,N)
        [観光地1[観光地1までの費用, 観光地2までの費用, ...], 
         観光地2[観光地1までの費用, 観光地2までの費用, ...], 
         ...
         ]
    x : dict
        {(0, 1): x(0,1),
         (0, 2): x(0,2),
         (0, 3): x(0,3),...
        }
        地点間の道を通るか否かの最適化問題の変数
    y : dict
        観光地を訪問するか否かの変数
    f : dict
        道順の変数(１以上の場合)
    """
    
    def __init__(self, I, a, b, pos, speed, c, m):
        """        
        parameters
        ----------
        I : array of int
            観光地名id(0以上の連続した数列)
        a : array of int or float
            観光地の満足度
        b : array of int or float
            観光地の滞在時間
        pos : ndarray of float
            [[x座標, y座標]_pos1, [x座標, y座標]_pos2]
            観光地の座標
        speed : float or int
            移動速度
        c : array of int
            観光地ごとの費用
        m : ndarray(N,N)
            [観光地1[観光地1までの費用, 観光地2までの費用, ...], 
             観光地2[観光地1までの費用, 観光地2までの費用, ...], 
             ...
             ]
        """
        self.I = I
        self.a = a
        self.b = b
        self.c = c
        self.d = [[compute_distance(pos_i[0], pos_i[1], pos_j[0], pos_j[1]) / speed 
                   for pos_i in pos] for pos_j in pos]
        self.m = m
        self.pos = pos
        
        
    def plot_map(self, x_min, x_max, y_min, y_max):
        """   
        マップの描画
        
        Parameters
        ----------
        x_min : float
            マップの描画範囲のx軸方向の最小値
        x_max : float
            マップの描画範囲のx軸方向の最大値
        x_min : float
            マップの描画範囲のy軸方向の最小値
        x_max : float
            マップの描画範囲のy軸方向の最大値
        """
        plt.figure(figsize=(15,15))

        for pos_i, name_i, a_i, b_i in zip(self.pos, map(str, self.I), 
                                           map(str, self.a), map(str, self.b)):
            plt.scatter(pos_i[0], pos_i[1])
            plt.annotate(f'name:{name_i}\nsatisf:{a_i}\nstay:{b_i}',
                         xy=(pos_i[0], pos_i[1]))

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.grid()
        plt.show()
        
    def plot_route(self, x, x_min, x_max, y_min, y_max):
        """   
        経路の描画
        
        Parameters
        ----------
        x : dict
            self.solveで得られた解
        x_min : float
            マップの描画範囲のx軸方向の最小値
        x_max : float
            マップの描画範囲のx軸方向の最大値
        x_min : float
            マップの描画範囲のy軸方向の最小値
        x_max : float
            マップの描画範囲のy軸方向の最大値
        """
        
        plt.figure(figsize=(15,15))
        for pos_i, name_i, a_i, b_i in zip(self.pos, map(str, self.I), 
                                           map(str, self.a), map(str, self.b)):
            plt.scatter(pos_i[0], pos_i[1])
            plt.annotate(f'name:{name_i}\nsatisf:{a_i}\nstay:{b_i}',
                         xy=(pos_i[0], pos_i[1]))

        for i in I:
            for j in I:
                if i != j and x[i,j].value() == 1:
                    plt.annotate('', xy=pos[i], xytext=pos[j], 
                                 arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                                 headlength=10, connectionstyle='arc3',
                                                 facecolor='gray', edgecolor='gray'))

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.grid()
        plt.show()
    
    
    def formulate(self, start, T, C):
        """
        最適化問題を定式化
        
        Parameters
        ----------
        start : int
            初期地点
        T : float
            旅行時間
        C : int
            旅費
        
        Returns
        -------
        problem
            pulp形式の最適化問題
        
        """
        
        # 数理最適化問題（最大化）を宣言
        problem = pulp.LpProblem("problem", pulp.LpMaximize)

        # 変数を定義
        self.y = {}
        for i in self.I:
            self.y[i] = pulp.LpVariable(f'y_{i}', 0, 1, pulp.LpInteger)

        self.x = {}
        for i in self.I:
            for j in self.I:
                if i != j:
                    self.x[i,j] = pulp.LpVariable(f'x({i},{j})', 0, 1, pulp.LpInteger)

        self.f = {}
        for i in self.I:
            for j in self.I:
                if i != j:
                    self.f[i,j] = pulp.LpVariable(f'f({i},{j})', 0, len(self.I), pulp.LpInteger)

        # 目的関数
        objective = pulp.lpSum(self.a[i] * self.y[i] for i in self.I)
        problem += objective

        # 制約条件
        ## 時間制約
        problem += pulp.lpSum(self.d[i][j] * self.x[i,j] for i in self.I for j in self.I if i != j) +\
                    pulp.lpSum(self.b[i] * self.y[i] for i in self.I) <= T

        ## 費用制約
        problem += pulp.lpSum(self.m[i][j] * self.x[i,j] for i in self.I for j in self.I if i != j) +\
                    pulp.lpSum(self.c[i] * self.y[i] for i in self.I) <= C

        ## 観光地を訪れるのは各１回
        for i in self.I:
            problem += pulp.lpSum(self.x[i,j] for j in self.I if i != j) == self.y[i]

        for j in self.I:
            problem += pulp.lpSum(self.x[i,j] for i in self.I if i != j) == self.y[j]

        ## 部分巡回路を排除
        for i in self.I:
            if i == start:
                for j in self.I:
                    if i != j:
                        problem += self.f[i,j] == 0
                continue

            problem += pulp.lpSum(self.f[h,i] for h in self.I if i != h) + self.y[i] == pulp.lpSum(self.f[i,j] for j in self.I if i != j)

        for i in self.I:
            for j in self.I:
                if i != j:
                    problem += self.f[i,j] <= len(self.I) * self.x[i,j]

        ## スタート地点を必ず通るようにする
        problem += self.y[start] == 1
        
        return problem
    
    def solve(self, start, T, C, threads=4, timeLimit=1):
        """
        最適化問題を解く
        
        Attribute
        ---------
        start : int
            初期地点
        T : float
            旅行時間
        C : int
            旅費
        threads : int
            並列数
        timeLimit : int
            問題を解く制限時間
        
        Returns
        -------
        x
            最適化問題の解
        """
        problem = self.formulate(start, T, C)
        solver = pulp.PULP_CBC_CMD(threads=threads, timeLimit=timeLimit)
        result_status = problem.solve(solver)
        
        # 実行可能解が存在している
        if result_status == 1:
            return self.x
        else:
            print("実行可能解が存在しません")
            return False
    
    def show_route(self, start, T, C, threads=4, timeLimit=1):
        """
        解となる観光地idの順番を返す
        
        Attribute
        ---------
        start : int
            初期地点
        T : float
            旅行時間
        C : int
            旅費
        threads : int
            並列数
        timeLimit : int
            問題を解く制限時間
        
        Returns
        -------
        x : list
            観光地の巡回順
        """
        
        ans = self.solve(start, T, C, threads=4, timeLimit=1)

        if ans: 
            route_dict = {k: v.value() for k, v in filter(lambda v: v[1].value() >= 1, self.f.items())}
            route_dict = sorted(route_dict.items(), key=lambda x: x[1])
            route = [start] + [v[0][0] for v in route_dict]

            return route
        else:
            return start
        