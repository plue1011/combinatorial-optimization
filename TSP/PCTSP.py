import numpy as np
import matplotlib.pyplot as plt
import pulp


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
        self.d = [[np.linalg.norm(pos_i - pos_j, ord=2) / speed for pos_i in pos] for pos_j in pos]
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
        y = {}
        for i in self.I:
            y[i] = pulp.LpVariable(f'y_{i}', 0, 1, pulp.LpInteger)

        self.x = {}
        for i in self.I:
            for j in self.I:
                if i != j:
                    self.x[i,j] = pulp.LpVariable(f'x({i},{j})', 0, 1, pulp.LpInteger)

        f = {}
        for i in self.I:
            for j in self.I:
                if i != j:
                    f[i,j] = pulp.LpVariable(f'f({i},{j})', 0, len(self.I), pulp.LpInteger)

        # 目的関数
        objective = pulp.lpSum(self.a[i] * y[i] for i in self.I)
        problem += objective

        # 制約条件
        ## 時間制約
        problem += pulp.lpSum(self.d[i][j] * self.x[i,j] for i in self.I for j in self.I if i != j) +\
                    pulp.lpSum(self.b[i] * y[i] for i in self.I) <= T

        ## 費用制約
        problem += pulp.lpSum(self.m[i][j] * self.x[i,j] for i in self.I for j in self.I if i != j) +\
                    pulp.lpSum(self.c[i] * y[i] for i in self.I) <= C

        ## 観光地を訪れるのは各１回
        for i in self.I:
            problem += pulp.lpSum(self.x[i,j] for j in self.I if i != j) == y[i]

        for j in self.I:
            problem += pulp.lpSum(self.x[i,j] for i in self.I if i != j) == y[j]

        ## 部分巡回路を排除
        for i in self.I:
            if i == start:
                for j in self.I:
                    if i != j:
                        problem += f[i,j] == 0
                continue

            problem += pulp.lpSum(f[h,i] for h in self.I if i != h) + y[i] == pulp.lpSum(f[i,j] for j in self.I if i != j)

        for i in self.I:
            for j in self.I:
                if i != j:
                    problem += f[i,j] <= len(self.I) * self.x[i,j]

        ## スタート地点を必ず通るようにする
        problem += y[start] == 1
        
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
        
        return self.x
        