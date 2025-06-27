import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize, newton


def mix_ideal(c1, c2, alpha):
    c_mix_ideal_rev = alpha / c1 + (1 - alpha) / c2
    return 1 / c_mix_ideal_rev


def mix_non_ideal(c1, c2, c_mix, alpha, beta):
    x1 = find_x1(alpha=alpha, c1=c1, c2=c2, c_mix=c_mix)
    f1 = calc_f1(beta, x1)
    f2 = calc_f2(beta, x1)
    c_mix_nonideal_rev = alpha / (f1 * c1) + (1 - alpha) / (f2 * c2)
    return 1 / c_mix_nonideal_rev


def calc_f1(beta, x1):
    f1 = np.exp(beta * (1 - x1) ** 2)
    return f1


def calc_f2(beta, x1):
    f2 = np.exp(beta * x1**2)
    return f2


def rubingh_equation(x1, alpha, c1, c2, c_mix):
    term1 = x1**2 * np.log((alpha * c_mix) / (x1 * c1))
    term2 = (1 - x1) ** 2 * np.log(((1 - alpha) * c_mix) / ((1 - x1) * c2))
    return term1 - term2



def find_x1(alpha, c1, c2, c_mix):
    def equation(x1):
        return rubingh_equation(x1, alpha, c1, c2, c_mix)
    
    if c_mix > mix_ideal(c1, c2, alpha):
        c_mix = mix_ideal(c1, c2, alpha)

    # Ensure the bracket is within (0,1) to avoid division by zero
    result = root_scalar(equation, bracket=[1e-6, 1 - 1e-6], method="brentq")
    if result.converged:
        x1 = result.root
        return result.root


def calculate_beta(x1, alpha, c1, c_mix):
    gamma1 = (alpha * c_mix) / (x1 * c1)
    beta = np.log(gamma1) / ((1 - x1) ** 2)
    return beta


def objective(beta, alpha_list, c_mix_list, c1, c2):
    # Remove first and last elements (pure components)
    alpha_list = alpha_list[1:-1]
    c_mix_list = c_mix_list[1:-1]
    residuals = []
    for alpha, c_mix in zip(alpha_list, c_mix_list):
            x1 = find_x1(alpha, c1, c2, c_mix)
            if x1 == np.nan:
                print("re")
                c_mix = mix_ideal(c1=c1, c2=c2, alpha=alpha) 
                x1 = find_x1(alpha, c1, c2, c_mix) 
            beta_calc = calculate_beta(x1, alpha, c1, c_mix)
            residuals.append((beta_calc - beta) ** 2)
    return np.sum(residuals)

def fit_beta(alpha_obs, c_mix_obs):
    c1 = c_mix_obs[-1]
    c2 = c_mix_obs[0]
    initial_beta = 0.0

    result = minimize(
        objective,
        initial_beta,
        args=(alpha_obs, c_mix_obs, c1, c2),
        method="Nelder-Mead",
    )

    if result.success:
        fitted_beta = result.x[0]
        print(f"Fitted interaction parameter β: {fitted_beta:.4f}")
    else:
        print("Optimization failed.")
        fitted_beta = initial_beta  # Fallback to initial guess

    return fitted_beta

if __name__ == "__main__":

    # Example data
    alpha_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # c_mix_list = [0.08,0.12,0.14,0.19,0.24,0.57]  # Experimental CMCs
    c_mix_list = [0.04,0.03,0.05,0.07,0.13,0.26]
    c1 = c_mix_list[-1] # alpha 1
    c2 = c_mix_list[0] # alpha 2

    # alpha_list = alpha_list[1:-1]
    # c_mix_list = c_mix_list[1:-1]

    # for alpha, c_mix in zip(alpha_list, c_mix_list):
    #     x1 = find_x1(alpha=alpha, c1=c1, c2=c2, c_mix=c_mix)
        # beta = calculate_beta(x1=x1, alpha=alpha, c1=c1, c_mix=c_mix)


    fitted_beta = fit_beta(alpha_obs=alpha_list, c_mix_obs=c_mix_list)

    # alphas_fit = np.linspace(0.01, 0.99, 100)

    # cmc_ideal_list = []
    # cmc_non_ideal_list = []
    # for alpha_fit in alphas_fit:
    #     cmc_ideal = mix_ideal(c1=c1, c2=c2, alpha=alpha_fit)
    #     cmc_ideal_list.append(cmc_ideal)
    #     cmc_non_ideal = mix_non_ideal(
    #         c1=c1, c2=c2, c_mix=alpha_fit, alpha=alpha_fit, beta=fitted_beta
    #     )
    #     cmc_non_ideal_list.append(cmc_non_ideal)

    # plt.figure(figsize=(3, 3))

    # plt.plot(
    #     alphas_fit,
    #     cmc_ideal_list,
    #     label="Ideal",
    #     color="black",
    #     alpha=0.5,
    # )
    # plt.plot(
    #     alphas_fit,
    #     cmc_non_ideal_list,
    #     label="Non-ideal",
    #     color="black",
    #     linestyle="--",
    #     alpha=0.5,
    # )
    # plt.scatter(alpha_list, c_mix_list)
    # plt.show()
