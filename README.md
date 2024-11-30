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
    
    
