# OpenDrawer exploratory depth-rule verification

Rules: episode median `< 0.22`; fraction with depth-change ratio `< 0.1` greater than `0.14`.

rule,n_episodes,failure_n,success_n,tp,tn,fp,fn,accuracy,sensitivity_failure_recall,specificity_success_recall,failure_precision
median_rule_failure,20,8,12,8,12,0,0,1.0,1.0,1.0,1.0
low_fraction_rule_failure,20,8,12,8,12,0,0,1.0,1.0,1.0,1.0


metric,failure_mean,failure_median,success_mean,success_median,failure_high_auc,mann_whitney_p
episode_median,0.125,0.13,0.3625,0.37,1.0,0.00023842135787542732
low_ratio_fraction,0.42075892857142855,0.4017857142857143,0.03316925369612588,0.031754032258064516,1.0,0.00023455467501049587


Both rules are evaluated retrospectively on complete episodes and use no operational onset. They are not cross-validated and the perfect separation on this 20-episode sample must not be treated as a generalization result.

Closest threshold margins: 
- median: failure minimum margin = 0.0300; success minimum margin = 0.0500
- low-fraction: failure minimum margin = 0.0207; success minimum margin = 0.0224
