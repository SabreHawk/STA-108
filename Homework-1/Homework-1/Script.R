# Read Data 
tmp_data = read.table("weight_full.txt", header = T);

x_data = tmp_data$height;
y_data = tmp_data$weight;


# calculate values
data_num = lengths(tmp_data[1]);
mean_height = mean(tmp_data$height)
mean_weight = mean(tmp_data$weight)
cat("mean_x : ",mean_height,'\n')
cat("mean_y : ",mean_weight,'\n')

# plot 
plot(x = x_data ,y = y_data, xlab = "Height", ylab = "Weight");
reg = lm(y_data ~ x_data);
abline(reg);
points(mean_height, mean_weight,cex=2,col="blue");
b1_up = 0;
q2 = 0;
q3 = 0;
q4 = 0;
for (i in 1:data_num) {
    b1_up = b1_up + (tmp_data$height[i] - mean_height) * (tmp_data$weight[i] - mean_weight);
    q2 = q2 + (x_data[i] - mean_height) * y_data[i];
    q3 = q3 + x_data[i] * (y_data[i] - mean_weight);
    q4 = q4 + x_data[i] * y_data[i];
}

b1_down = 0;x
for (i in 1:data_num) {
    b1_down = b1_down + (tmp_data$height[i] - mean_height) ^ 2;
}
b1 = b1_up / b1_down;
cat("b1 : ",b1,'\n')
b0 = mean_weight - b1 * mean_height;
cat("b0 : ",b0,'\n')

cat("q1 : ", b1_up, '\n');
cat("q2 : ", q2, '\n');
cat("q3 : ", q3, '\n');
cat("q4 : ", q4, '\n');

