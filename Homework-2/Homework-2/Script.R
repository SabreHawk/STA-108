# Read Data 
tmp_data = read.table("crimerate.txt", header = T);
#Reg
data_num = lengths(tmp_data[1]);
mean_rate = mean(tmp_data$rate);
mean_diploma = mean(tmp_data$pct.diploma);
b1_up = 0;
b1_down = 0;
for (i in 1:data_num) {
    b1_up = b1_up + (tmp_data$pct.diploma[i] - mean_diploma) * (tmp_data$rate[i] - mean_rate);
    b1_down = b1_down + (tmp_data$pct.diploma[i] - mean_diploma) ^ 2;
}
b1 = b1_up / b1_down;
b0 = mean_rate - mean_diploma* b1;
print(b1);
print(b0);
MSE = 0;
SE_b1_down = 0;
SE_b0_rd = 0;
for (i in 1:data_num) {
    MSE = MSE + (tmp_data$rate[i] - (b0 + b1 * tmp_data$pct.diploma[i])) ^ 2;
    SE_b1_down = SE_b1_down + (tmp_data$pct.diploma[i] - mean_diploma) ^ 2;
    SE_b0_rd = SE_b0_rd + (tmp_data$pct.diploma[i] - mean_diploma) ^ 2;
}
MSE = MSE / (data_num - 2);
cat("MES :", MSE, '\n');
SE_b1 = sqrt(MSE / SE_b1_down);
SE_b0 = sqrt(MSE * (1 / data_num + mean_diploma ^ 2 / SE_b0_rd));

#2.e
critical_value_0.025_83 = 1.989;
abs_T = mean_diploma / sqrt(MSE / data_num)
print(abs_T)
critical_value_82_0.975 = 1.989
right_boundary_e = b1 - critical_value_82_0.975 * SE_b1;
left_boundary_e = b1 + critical_value_82_0.975 * SE_b1;
cat("CI - b1 : [", left_boundary_e, " , ", right_boundary_e, "]", '\n');

#2.f
#mean_x = 80;
#critical_value_83_0.995 = 2.636
#right_boundary_f = mean_x - critical_value_83_0.995 * sqrt(MSE / data_num);
#left_boundary_f = mean_x + critical_value_83_0.995 * sqrt(MSE / data_num);
#cat("CI - mean : [", left_boundary_f, " , ", right_boundary_f, "]", '\n');
print(pt(0.995,84))
tmp_v = b0 + b1 * 80;

tmp_vv = sqrt(MSE * (1 / data_num + (mean_diploma - 80) ^ 2 / SE_b0_rd));
cat("CI - mean : [", tmp_v - tmp_vv, " , ", tmp_v+tmp_vv, "]", '\n');


plot(x = tmp_data$pct.diploma, y = tmp_data$rate, xlab = "rate", ylab = "pct.diploma");
reg = lm(tmp_data$rate ~ tmp_data$pct.diploma);
abline(reg);
