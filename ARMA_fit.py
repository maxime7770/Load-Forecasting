

def duple(n):   # on va renvoyer des couples (i,j) pour tenter de fit un arma (i,j)
    res =[]
    for i in range(0,n):
        for j in  range(0,n):
            res.append((i,j))
    return res



def ARMAfit(filename,n):
    S = pd.read_csv(filename)
    conso = S['Consommation'].values()
    M = {}
    for p,q in models:
        try:
        #fit an arma (for now without trend)
            ft =  ARMA(conso,order=(p,q)).fit()
            M[p,q] = ft
        except ValueError:
            print()
    AIC = pd.DataFrame( [(m,ft.aic) for m,ft in M.items()],columns=['model','AIC'] ) 
    AIC = AIC.assign(dAIC=(AIC.AIC-AIC.AIC.min()))
    k = np.argmin(AIC['dAIC'])
    p,q = AIC.model[k]
    print('le meilleure modèle est un arma' )
    print(p,q)
    r = M[p,q].resid
    print(la p-value est)
    print(acorr_ljungbox(R,lags=[p+q+1],model_df=p+q,return_df=True)['lb_pvalue'])