# Historical Variance Decomposition VAR Model

## Parameters

    def extract_VAR_parameters(results, data, endog_var, exog_var=None, nlag_ex=0):

        sigma = results.sigma_u
        coefs = results.coefs
        nvar = coefs.shape[1]
        nlag = coefs.shape[0]
        companion_matrix = np.zeros((nvar * nlag, nvar * nlag))
        companion_matrix[:nvar, :] = np.hstack(coefs)
        if nlag > 1:
            companion_matrix[nvar:, :-nvar] = np.eye(nvar * (nlag - 1))
        Fcomp = companion_matrix
    
    
        const = results.params[:nvar]
    
        intercept = results.intercept.reshape(-1, 1)  # Convertir a una matriz columna
        F = np.hstack([intercept] + [results.coefs[i] for i in range(nlag)]).T
    
        nobs = results.nobs
    
        X = []
        for lag in range(1, nlag + 1):
            X.append(data[endog_var].shift(lag).iloc[nlag:])
        X = pd.concat(X, axis=1).dropna().values  # Concatenar y eliminar valores NaN
    
        if exog_var is not None:
            X_exog = []
            for lag in range(nlag_ex + 1):  # Incluir el rezago 0
                X_exog.append(data[exog_var].shift(lag).iloc[nlag:])
            X_exog = pd.concat(X_exog, axis=1).dropna().values
            nvar_ex = len(exog_var)  # Ajustar con el número real de variables exógenas
        else:
            X_exog = np.array([])  # Si no hay exógenas, inicializar vacío
            nvar_ex = 0
    
        Y = data[endog_var].iloc[nlag:].values  # Eliminar los primeros nlag valores para alinearlos con X
    
        nvar_ex = pd.DataFrame([columnas_exogenas]).shape[1] if exog_var is not None else 0
    
        var_params = {
            "sigma": sigma,
            "Fcomp": Fcomp,
            #"Fcompexog": Fcompexog,
            "const": const,
            "F": F,
            #"F_exog": F_exog,
            "nvar": nvar,
            "nvar_ex": nvar_ex,  # Número de variables exógenas
            "nvarXeq": nvar * nlag,
            "nlag": nlag,
            "nlag_ex": nlag_ex,  # Rezagos de las exógenas
            "Y": Y,
            "X": X,
            "X_exog": X_exog,  # Matriz de variables exógenas con rezagos
            "nobs": nobs
        }
    
        return var_params
    
    def recover_B_matrix(sigma, VARopt):
    
        if VARopt['ident'] == 'short':
            try:
                # Descomposición de Cholesky
                B = np.linalg.cholesky(sigma).T
            except np.linalg.LinAlgError:
                raise ValueError("La matriz VCV no es positiva definida.")
            
            return B

## Perform Historical Variance Decomposition

    def historical_decomposition(var_params, residuals, HDtrendVal = False, HDtrendValsq = False):

        
        sigma = var_params["sigma"]
        Fcomp = var_params["Fcomp"]
        const = np.array(var_params["const"]).flatten()[:var_params["nvar"]]  # Ajustar dimensiones
        F = var_params["F"]
        nvar = var_params["nvar"]
        nlag = var_params["nlag"]
        nobs = var_params["nobs"]
        X = var_params["X"]
        nvarXeq = var_params["nvarXeq"]
        nvar_ex = var_params["nvar_ex"]
        
        VARopt = {'ident': 'short'}
        B = recover_B_matrix(sigma, VARopt)
    
        eps = np.linalg.inv(B) @ residuals.T
        eps = np.array(eps)  
    
        B_big = np.zeros((nvarXeq, nvar))
        B_big[:nvar, :] = B
        Icomp = np.hstack([np.eye(nvar), np.zeros((nvar, (nlag - 1) * nvar))])
    
        HDshock_big = np.zeros((nvarXeq, nobs + 1, nvar))
        HDshock = np.zeros((nvar, nobs + 1, nvar))
    
        for j in range(nvar):  # Para cada variable
            eps_big = np.zeros((nvar, nobs + 1))
            eps_big[j, 1:] = eps[j, :]  # Alinear errores estructurales
            for i in range(1, nobs + 1):
                
                HDshock_big[:, i, j] = B_big @ eps_big[:, i] + Fcomp @ HDshock_big[:, i - 1, j]
                HDshock[:, i, j] = Icomp @ HDshock_big[:, i, j]
    
        HDinit_big = np.zeros((nvarXeq, nobs + 1))
        HDinit = np.zeros((nvar, nobs + 1))
        HDinit_big[:, 0] = X[0, :].T
        HDinit[:, 0] = Icomp @ HDinit_big[:, 0]
        for i in range(1, nobs + 1):
            HDinit_big[:, i] = Fcomp @ HDinit_big[:, i - 1]
            HDinit[:, i] = Icomp @ HDinit_big[:, i]
    
        HDconst_big = np.zeros((nvarXeq, nobs + 1))
        HDconst = np.zeros((nvar, nobs + 1))
        if len(const) > 0:
            CC = np.zeros((nvarXeq, 1))
            CC[:nvar, :] = const.reshape(-1, 1)
            for i in range(1, nobs + 1):
                HDconst_big[:, i] = CC[:, 0] + Fcomp @ HDconst_big[:, i - 1]
                HDconst[:, i] = Icomp @ HDconst_big[:, i]
    
        if HDtrendVal == True:
            HDtrend_big = np.zeros((nvarXeq, nobs + 1))
            HDtrend = np.zeros((nvar, nobs + 1))
            if F.shape[1] > 1:
                TT = np.zeros((nvarXeq, 1))
                TT[:nvar, :] = F[:nvar, 1].reshape(-1, 1)  # Asegurar que coincidan dimensiones
                for i in range(1, nobs + 1):
                    HDtrend_big[:, i] = TT[:, 0] * (i - 1) + Fcomp @ HDtrend_big[:, i - 1]
                    HDtrend[:, i] = Icomp @ HDtrend_big[:, i]
        else:
            HDtrend = 0

        if HDtrendValsq == True:
            HDtrend2_big = np.zeros((nvarXeq, nobs + 1))
            HDtrend2 = np.zeros((nvar, nobs + 1))
            if F.shape[1] > 2:
                TT2 = np.zeros((nvarXeq, 1))
                TT2[:nvar, :] = F[:nvar, 2].reshape(-1, 1)
                for i in range(1, nobs + 1):
                    HDtrend2_big[:, i] = TT2[:, 0] * ((i - 1) ** 2) + Fcomp @ HDtrend2_big[:, i - 1]
                    HDtrend2[:, i] = Icomp @ HDtrend2_big[:, i]
        else:
            HDtrend2 = 0
    
        HDexo_big = np.zeros((nvarXeq, nobs + 1, max(1, nvar_ex)))  
        HDexo = np.zeros((nvar, nobs + 1, max(1, nvar_ex)))  
    
        if nvar_ex > 0 and var_params["X_exog"].size > 0:
            exog_start_idx = nvar * nlag  
            for ii in range(nvar_ex):
                if ii >= var_params["X_exog"].shape[1]:
                    continue  
                VARexo = var_params["X_exog"][:, ii]  
                EXO = np.zeros((nvarXeq, 1))
                if exog_start_idx + ii < F.shape[1]:  
                    EXO[:nvar, 0] = F[:nvar, exog_start_idx + ii]  
                for i in range(1, nobs + 1):
                    HDexo_big[:, i, ii] = EXO[:, 0] * VARexo[i - 1] + Fcomp @ HDexo_big[:, i - 1, ii]
                    HDexo[:, i, ii] = Icomp @ HDexo_big[:, i, ii]
        else:
            HDexo = np.zeros((nvar, nobs + 1, 1))  
    
        HDtotal = HDinit + HDconst + HDtrend + HDtrend2 + HDshock.sum(axis=2) + HDexo.sum(axis=2)
    
        return HDtotal, HDshock, HDinit, HDconst, HDtrend, HDtrend2, HDexo
    
## Validate if contribution add to the observed $Y_t$

    def validate_decomposition(var_params, HDtotal):
        Y_observed = var_params["Y"]
    
        diff = np.abs(HDtotal[:, 1:].T - Y_observed)
        if not np.allclose(diff, 0, atol=1e-5):
            raise AssertionError(f"Contributions do not add up properly. Max error: {np.max(diff)}. The function above still not handle properly exogenous variables")
        print("Contributions add up properly.")
        return True
    

