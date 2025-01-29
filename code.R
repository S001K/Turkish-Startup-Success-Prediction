library (MASS)
library (ISLR)
library (class)
library (boot)
library (tree)
library (randomForest)
library (rpart)
library (ggplot2)
library (xgboost)
library (gbm)
library (e1071)
library (ROCR)
library (neuralnet)
library(pheatmap)
library(ggplot2)
library(reshape2)
library(caret)
#-------------------------------------------------------

# Reading Data set
data <- read.csv("turkishCF.csv", sep = ";")

# Dropping unrelated columns. These columns give no logical meaning to the relationship with the target value so for avoiding unnecessary bias into our model 
# we decided to drop these columns.
data <- data[, !(names(data) %in% c("id", "proje_adi","proje_sahibi","proje_baslama_tarihi","proje_bitis_tarihi","proje_aciklamasi"))]

# Dropping "konum" column. Because after trying one-hot encoding on our data set, especially this "konum" columns (with too many unique values) caused High Cardinality
# in our data set. We solved this issue by using the grouped option which was already included in the data set ("bölge" column). 
data <- data[, !(names(data) %in% c("konum"))]

#-------------------------------------------------------

# correlation
# install.packages("pheatmap")

# Converting "destek_orani" column into integer values.
data$destek_orani <- as.numeric(gsub("%", "", data$destek_orani))



# Example data: Select numerical columns
numeric_data <- data[, sapply(data, is.numeric)]

# Example data: Select categorical columns
categoric_data <- data[, sapply(data, function(col) !is.numeric(col))]






# Compute the correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs", method = "pearson")


# --- Pheat map ---
pheatmap(cor_matrix, 
         display_numbers = TRUE, 
         color = colorRampPalette(c("red", "white", "blue"))(50), 
         main = "Correlation Heatmap (pheatmap)")


# drop highly correlated features (toplanan_tutar, sm_sayisi maybe?)
data <- data[, !(names(data) %in% c("toplanan_tutar"))]

#-------------------------------------------------------

# Converting "destek_orani" column into integer values.
data$destek_orani <- as.numeric(gsub("%", "", data$destek_orani))

# Detecting Missing values
missing_rows_count <- sum(apply(data, 1, function(row) any(is.na(row))))
print(missing_rows_count)
missing_rows_count <- sum(rowSums(is.na(data)) > 0)
print(missing_rows_count)


#-------------------------------------------------------

# Categorical
data$yil <- as.character(data$yil)

#-------------------------------------------------------

# Changing to NA for "bolge" feature for values "","belirsiz","genel"
data$bolge[data$bolge == ""] <- NA
data$bolge[data$bolge == "genel"] <- NA
data$bolge[data$bolge == "belirsiz"] <- NA


# Example data: Select numerical columns
numeric_data <- data[, sapply(data, is.numeric)]

# Example data: Select categorical columns
categoric_data <- data[, sapply(data, function(col) !is.numeric(col))]

######## Visualizations #######



######## Visualizations #######



# Load required libraries
library(ggplot2)
library(gridExtra)
library(dplyr)

# Identify numerical and categorical features

numeric_data <- cbind(numeric_data, basari_durumu = data$basari_durumu)
numeric_data$basari_durumu <- as.numeric(as.factor(numeric_data$basari_durumu)) - 1



numerical_features <- names(numeric_data)
print(numerical_features)
categorical_features <- names(categoric_data)
print(categorical_features)

# 
# # Create plots for numerical features (histograms) with dynamic axis limits
# numerical_plots <- lapply(numerical_features, function(col) {
#   # Calculate x-axis range
#   x_range <- range(numeric_data[[col]], na.rm = TRUE)
#   
#   # Create the histogram with dynamic x and y axis limits
#   ggplot(numeric_data, aes_string(x = col)) +
#     geom_histogram(binwidth = 2, fill = "#8D31BF", color = "white") +
#     labs(title = paste("Histogram of", col), x = col, y = "Count") +
#     theme_minimal() +
#     scale_x_continuous(limits = x_range) + 
#     scale_y_continuous(expand = expansion(mult = c(0, 0.1))) # Adjust to fit bars nicely
# })
# 
# 
# # Create plots for categorical features (bar plots with hue)
# categorical_plots <- lapply(categorical_features, function(col) {
#   ggplot(categoric_data, aes_string(x = col, fill = "basari_durumu")) +
#     geom_bar(position = "dodge") +
#     labs(title = paste("Bar Plot of", col, "by Target"), x = col, y = "Count") +
#     theme_minimal()
# })
# 
# # Create normalized bar plots for categorical features
# normalized_categorical_plots <- lapply(categorical_features, function(col) {
#   ggplot(categoric_data %>% 
#            group_by_at(c("basari_durumu", col)) %>% 
#            summarise(count = n(), .groups = "drop") %>% 
#            group_by_at(col) %>% 
#            mutate(prop = count / sum(count)),
#          aes_string(x = col, y = "prop", fill = "basari_durumu")) +
#     geom_bar(stat = "identity", position = "dodge") +
#     labs(title = paste("Normalized Bar Plot of", col, "by Target"), x = col, y = "Proportion") +
#     theme_minimal()
# })
# 
# 
# # Set limits for the number of plots
# max_plots <- 20  # Adjust this value as needed
# 
# # Limit the number of plots for each category
# numerical_plots_limited <- head(numerical_plots, max_plots)
# categorical_plots_limited <- head(categorical_plots, max_plots)
# normalized_categorical_plots_limited <- head(normalized_categorical_plots, max_plots)

# # Combine and arrange the plots
# # grobs = c(numerical_plots_limited, categorical_plots_limited, normalized_categorical_plots_limited),
# grid.arrange(
#   grobs = c(numerical_plots_limited),
#   ncol = 5
# )

library(ggplot2)
library(gridExtra)

# Specify the range of numerical and categorical plots you want to display
numerical_range <- 1:20
categorical_range <- 1:12

# Create plots for numerical features (Violin Plots)
numerical_plots <- lapply(numerical_features[numerical_range], function(col) {
  ggplot(data, aes_string(y = col, x = "1")) +  # Dummy x-axis to create a single violin
    geom_violin(fill = "#E88E50", color = "#7186E0") +
    labs(title = paste("Violin Plot of", col), y = col, x = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())  # Hide x-axis
})

# Create plots for categorical features (Normalized Stacked Bar Plots)
categorical_plots <- lapply(categorical_features[categorical_range], function(col) {
  ggplot(categoric_data, aes_string(x = col, fill = "basari_durumu")) +
    geom_bar(aes(y = ..prop.., group = 1), position = "stack") +
    labs(
      title = paste("Normalized Stacked Bar Plot of", col), 
      x = col, 
      y = "Proportion", 
      fill = "Basari Durumu"  # Add legend title
    ) +
    theme_minimal() +
    scale_y_continuous(labels = scales::percent) +  # Display y-axis as percentage
    scale_fill_manual(
      values = c("#E88E50", "#7186E0"),  # Custom colors
      labels = c("Basarisiz", "Basarili")  # Add meaningful labels
    )
})

# Combine and arrange the plots in a grid
grid.arrange(
  grobs = c(numerical_plots, categorical_plots),
  ncol = 4  # Adjust the number of columns as needed
)

# Print feature information
print("Numerical Features:")
print(numerical_features[numerical_range])

print("Categorical Features:")
print(categorical_features[categorical_range])





library(ggplot2)
library(gridExtra)

# Specify the range of numerical and categorical plots you want to display
numerical_range <- 1:20
categorical_range <- 1:12

# Create plots for numerical features (Violin Plots with Quartiles and Mini Boxplots)
numerical_plots <- lapply(numerical_features[numerical_range], function(col) {
  ggplot(data, aes_string(y = col, x = "1")) +  # Dummy x-axis to create a single violin
    geom_violin(fill = "#E88E50", color = "#7186E0", alpha = 0.8) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +  # Add mini boxplot
    stat_summary(fun = median, geom = "point", shape = 21, size = 3, fill = "white") +  # Highlight median
    labs(title = paste("Violin Plot with Boxplot of", col), y = col, x = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())  # Hide x-axis
})

# Create plots for categorical features (Normalized Bar Plots)
categorical_plots <- lapply(categorical_features[categorical_range], function(col) {
  ggplot(categoric_data, aes_string(x = col, fill = "basari_durumu")) +
    geom_bar(aes(y = ..count../sum(..count..)), position = "dodge") +  # Normalize by total count
    labs(
      title = paste("Normalized Bar Plot of", col), 
      x = col, 
      y = "Proportion", 
      fill = "Basari Durumu"  # Add legend title
    ) +
    theme_minimal() +
    scale_y_continuous(labels = scales::percent) +  # Display y-axis as percentage
    scale_fill_manual(values = c("#E88E50", "#7186E0"))  # Custom colors
})

# Combine and arrange the plots in a grid
grid.arrange(
  grobs = c(categorical_plots),
  ncol = 4  # Adjust the number of columns as needed
)

# Print feature information
print("Numerical Features:")
print(numerical_features[numerical_range])

print("Categorical Features:")
print(categorical_features[categorical_range])







# Bar Plot (Normalized by Proportion)
normalized_bar_plot <- ggplot(categoric_data, aes_string(x = "bolge", fill = "basari_durumu")) +
  geom_bar(aes(y = ..count../sum(..count..)), position = "dodge") +  # Normalize by total count
  labs(title = paste("Normalized Bar Plot of", "bolge"), 
       x = "bolge", 
       y = "Proportion") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent) +  # Display y-axis as percentage
  scale_fill_manual(values = c("#E88E50", "#7186E0"))  # Custom colors

# Print the second plot
print(normalized_bar_plot)







######## Visualizations #######



######## Visualizations #######








#-------------------------------------------------------

# Calculate the mode for the categorical columns
calculate_mode <- function(column) {
  unique_vals <- unique(column)
  unique_vals <- unique_vals[!is.na(unique_vals)]  # Dropping NA values
  mode_val <- unique_vals[which.max(tabulate(match(column, unique_vals)))]
  return(mode_val)
}

for (col in names(data)) {
  if (any(is.na(data[[col]]))) {  # If the column has NA value
    if (is.numeric(data[[col]])) {
      # fill the numerical columns by mean value
      mean_val <- mean(data[[col]], na.rm = TRUE)
      data[[col]][is.na(data[[col]])] <- mean_val
    } else {
      # fill the categorical columns by the mode value
      mode_val <- calculate_mode(data[[col]])
      data[[col]][is.na(data[[col]])] <- mode_val
    }
  }
}

#-------------------------------------------------------

# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 <- quantile(data$sm_takipci, 0.25)
Q3 <- quantile(data$sm_takipci, 0.75)

# Calculate the IQR
IQR <- Q3 - Q1

# Define the lower and upper bounds
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify outliers
outliers <- data$sm_takipci[data$sm_takipci < lower_bound | data$sm_takipci > upper_bound]
print(outliers)  # Print the outliers
print(length(outliers))

#-------------------------------------------------------

# Just to checking the process
# NA değerleri hariç tutarak maksimum değeri bul
max_sm_takipci <- max(data$sm_takipci, na.rm = TRUE)

# Maksimum değeri yazdır
print(max_sm_takipci)
# outlier 1st and 99th

#-------------------------------------------------------

# Changing the outliers by 1st and 99th percentiles
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    # Calculate the 1st and 99th percentiles
    percentile_1 <- quantile(data[[col]], 0.01, na.rm = TRUE)
    percentile_99 <- quantile(data[[col]], 0.99, na.rm = TRUE)
    
    # Changing the outliers
    data[[col]] <- ifelse(data[[col]] < percentile_1, percentile_1, data[[col]])
    data[[col]] <- ifelse(data[[col]] > percentile_99, percentile_99, data[[col]])
  }
}

# Checking the summary of process
summary(data)

# IF you want to checking the new outlier values
outlier_counts <- sapply(data, function(col) {
  if (is.numeric(col)) {
    percentile_1 <- quantile(col, 0.01, na.rm = TRUE)
    percentile_99 <- quantile(col, 0.99, na.rm = TRUE)
    sum(col < percentile_1 | col > percentile_99, na.rm = TRUE)
  } else {
    NA
  }
})

print(outlier_counts)

#-------------------------------------------------------

# Just for checking
# NA değerleri hariç tutarak maksimum değeri bul
max_sm_takipci <- max(data$sm_takipci, na.rm = TRUE)

# Maksimum değeri yazdır
print(max_sm_takipci)
# outlier 1st and 99th

#-------------------------------------------------------

# Define the normalization function
min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Normalize the entire data frame
data <- as.data.frame(lapply(data, function(x) {
  if (is.numeric(x)) {
    (x - min(x)) / (max(x) - min(x))
  } else {
    x  # Keep non-numeric columns as is
  }
}))

# View the result
print(data)

#-------------------------------------------------------

# "kac_projeye_abone" column has not any value so we dropped it

unique_count <- length(unique(data$kac_projeye_abone))
print(unique_count)

unique_list <- unique(data$kac_projeye_abone)
print(unique_list)


# Sütunu veri setinden kaldır
data <- data[, !(names(data) %in% c("kac_projeye_abone"))]

#-------------------------------------------------------

# Encoding the Features

non_numeric_columns <- names(data)[!sapply(data, is.numeric)]
print(non_numeric_columns)
typeof(non_numeric_columns)
length(non_numeric_columns)


# Convert the list of character vectors into a single vector
non_numeric_columns_single_vector <- unlist(non_numeric_columns)
print(non_numeric_columns_single_vector)
typeof(non_numeric_columns_single_vector)


# Install and load dummies package if not already installed
if (!require(fastDummies)) install.packages("fastDummies")

library(fastDummies)

# Create dummy variables
data <- dummy_cols(data, select_columns = non_numeric_columns_single_vector, remove_first_dummy = TRUE)

#-------------------------------------------------------

# Dropped the old "non_numeric_columns"
data <- data[, !(names(data) %in% c(non_numeric_columns))]

################
# Correcting column names turkish character to english

column_names <- colnames(data)
formatted_output <- paste0('("', paste(column_names, collapse = '", "'), '")')
print(formatted_output)

cat('("', paste(column_names, collapse = '", "'), '")', sep = "")

transposed_head <- data.frame(t(head(data)))
print(transposed_head)

new_column_names <- c("kac_proje_destekledi","kac_projenin_sahibi", "kac_proje_takiminda", "gun_sayisi", "video_uzunlugu", "gorsel_sayisi",
                      "sss", "guncellemeler", "yorumlar", "destekci_sayisi", "odul_sayisi", "ekip_kisi_sayisi", "sm_sayisi", "sm_takipci", 
                      "etiket_sayisi", "icerik_kelime_sayisi", "hedef_miktari", "destek_orani", "platform_adi_bulusum", "platform_adi_crowdfon", 
                      "platform_adi_fonbulucu", "platform_adi_fongogo", "platform_adi_ideanest", "kitle_fonlamasi_turu_odul", "kategori_dans_performans", 
                      "kategori_diger", "kategori_egitim", "kategori_film_video_fotograf", "kategori_gida_yeme_icme", "kategori_hayvanlar", 
                      "kategori_kultur_sanat", "kategori_moda", "kategori_muzik", "kategori_saglik_guzellik", "kategori_sosyal_sorumluluk", 
                      "kategori_spor", "kategori_tasarim", "kategori_teknoloji", "kategori_turizm", "kategori_yayincilik", "fon_sekli_ya_hep_ya_hic", 
                      "proje_sahibi_cinsiyet_erkek", "proje_sahibi_cinsiyet_kadin", "bolge_dogu", "bolge_ege", "bolge_guneydogu", "bolge_ic_anadolu", 
                      "bolge_karadeniz", "bolge_marmara", "yil_2012", "yil_2013", "yil_2014", "yil_2015", "yil_2016", "yil_2017", "yil_2018", "yil_2019", 
                      "yil_2020", "yil_2021", "tanitim_videosu_yok", "web_sitesi_yok", "sosyal_medya_yok", "basari_durumu_basarisiz")

colnames(data) <- new_column_names
column_names <- colnames(data)
cat('("', paste(column_names, collapse = '", "'), '")', sep = "")

# Feature Importance
library(FSelector)

result <- information.gain(basari_durumu_basarisiz ~ ., data = data)
typeof(result)
str(result)
print(result)


# result veri çerçevesinin 'attr_importance' sütununu ve sütun isimlerini birleştirme
result_df <- data.frame(
  Feature = rownames(result),          # Özellik isimlerini al
  InformationGain = result$attr_importance  # Bilgi kazancı değerlerini al
)

# Bilgi kazancı değerlerine göre sıralama (azalan)
sorted_result_df <- result_df[order(result_df$InformationGain, decreasing = TRUE), ]

# Plot the information gain chart
ggplot(sorted_result_df, aes(x = reorder(Feature, InformationGain), y = InformationGain, fill = InformationGain)) +
  geom_bar(stat = "identity", color = "black") +         # Create a bar chart
  scale_fill_gradient(low = "blue", high = "red") +     # Color gradient
  coord_flip() +                                        # Flip axes for better readability
  labs(
    title = "Information Gain by Feature",
    x = "Features",
    y = "Information Gain",
    fill = "Gain"
  ) +
  theme_minimal() +                                     # Minimal theme for a clean look
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),  # Center the title
    axis.text = element_text(size = 12),               # Adjust axis text size
    axis.title = element_text(size = 14)               # Adjust axis title size
  )

# Eşik değeri belirleme
threshold <- 0.01

# Bilgi kazancı 0.01'den büyük olan özellikleri seçme
selected_features1 <- rownames(result)[result$attr_importance > threshold]

# Seçilen özellikleri yazdırma
print(selected_features1)
str(selected_features1)

# Eşik değeri belirleme
threshold <- 0.03

# Bilgi kazancı 0.03'den büyük olan özellikleri seçme
selected_features2 <- rownames(result)[result$attr_importance > threshold]

# Seçilen özellikleri yazdırma
print(selected_features2)
str(selected_features2)

# Eşik değeri belirleme
threshold <- 0.15

# Bilgi kazancı 0.15'den büyük olan özellikleri seçme
selected_features3 <- rownames(result)[result$attr_importance > threshold]


# Seçilen özellikleri yazdırma
print(selected_features3)
str(selected_features3)

# group1'e göre dataframe oluşturma
data_group1 <- data[, selected_features1]
data_group1 <- cbind(data_group1, basari_durumu_basarisiz = data$basari_durumu_basarisiz)
# group2'ye göre dataframe oluşturma
data_group2 <- data[, selected_features2]
data_group2 <- cbind(data_group2, basari_durumu_basarisiz = data$basari_durumu_basarisiz)
# group3'e göre dataframe oluşturma
data_group3 <- data[, selected_features3]
data_group3 <- cbind(data_group3, basari_durumu_basarisiz = data$basari_durumu_basarisiz)
# 4. Adım: Yeni veri kümesi olarak bu kesilen dataframe'leri inceleyebilirsiniz
# Örneğin, group1'e ait sütunları yazdıralım
print(head(data_group3))




# Hedef değişken dağılımını görmek için pie chart
target_distribution <- table(data$basari_durumu_basarisiz)  # Hedef değişkenin dağılımı

# Pie chart çizme
pie(target_distribution, 
    main = "Hedef Değişken Dağılımı", 
    col = c("lightblue", "lightgreen", "lightcoral"), 
    labels = paste(names(target_distribution), "\n", target_distribution))

# Yüzde hesaplama
target_percentage <- round(target_distribution / sum(target_distribution) * 100, 1)

# Pie chart çizme ve yüzdeleri etiket olarak ekleme
pie(target_distribution, 
    main = "Hedef Değişken Dağılımı", 
    col = c("lightblue", "lightgreen", "lightcoral"), 
    labels = paste(names(target_distribution), "\n", target_percentage, "%"))

















######## KNN #######



######## KNN #######


# dataset without highly_correlated feauture "destek_orani"

data_whcf_group1 <- subset(data_group1, select = -c(destek_orani))
data_whcf_group2 <- subset(data_group2, select = -c(destek_orani))
data_whcf_group3 <- subset(data_group3, select = -c(destek_orani))


# Load necessary libraries
if (!require(class)) install.packages("class")  # For KNN
if (!require(rsample)) install.packages("rsample")  # For Cross-validation
if (!require(pROC)) install.packages("pROC")  # For ROC-AUC
library(class)
library(rsample)
library(pROC)

# Ensure data_whcf_group3 is available
# Replace this with your actual dataset loading if not available
if (!exists("data_whcf_group3")) {
  stop("data_whcf_group3 veri seti tanımlanmamış.")
}

# Ensure target variable is a factor
data_whcf_group3$basari_durumu_basarisiz <- as.factor(data_whcf_group3$basari_durumu_basarisiz)

# Stratified 10-fold cross-validation
set.seed(123)
split <- vfold_cv(data_whcf_group3, v = 10, strata = "basari_durumu_basarisiz")

# Initialize lists to store metrics
results <- list()
precision_list <- numeric(length(split$splits))
recall_list <- numeric(length(split$splits))
f1_list <- numeric(length(split$splits))
roc_auc_list <- numeric(length(split$splits))

# KNN model for each fold
for (i in 1:length(split$splits)) {
  # Split data into training and testing
  train_data <- analysis(split$splits[[i]])
  test_data <- assessment(split$splits[[i]])
  
  # Ensure the correct column index for the target variable
  target_col <- "basari_durumu_basarisiz"
  target_index <- which(colnames(train_data) == target_col)
  
  # KNN model (k = 5)
  knn_model <- knn(
    train = train_data[, -target_index],
    test = test_data[, -target_index],
    cl = train_data[[target_col]],
    k = 3
  )
  
  # Convert predictions to factors for compatibility
  predictions <- factor(knn_model, levels = levels(test_data[[target_col]]))
  actual <- test_data[[target_col]]
  
  # Calculate metrics
  confusion_matrix <- table(Predicted = predictions, Actual = actual)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  precision <- diag(confusion_matrix) / rowSums(confusion_matrix)
  recall <- diag(confusion_matrix) / colSums(confusion_matrix)
  # Calculate Specificity
  TN <- confusion_matrix[2, 2]  # True Negatives
  FP <- confusion_matrix[1, 2]  # False Positives
  specificity <- TN / (TN + FP)
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  # Handle cases where precision or recall might be NA
  precision <- ifelse(is.na(precision), 0, precision)
  recall <- ifelse(is.na(recall), 0, recall)
  f1 <- ifelse(is.na(f1), 0, f1)
  
  # Calculate ROC-AUC
  roc_obj <- tryCatch({
    roc(actual, as.numeric(predictions), levels = rev(levels(actual)))
  }, error = function(e) {
    return(NULL)
  })
  roc_auc <- if (!is.null(roc_obj)) auc(roc_obj) else NA
  
  # Store metrics
  results[[i]] <- accuracy
  precision_list[i] <- mean(precision)
  recall_list[i] <- mean(recall)
  f1_list[i] <- mean(f1)
  roc_auc_list[i] <- ifelse(is.na(roc_auc), 0, roc_auc)
  
}

# Average metrics across folds
mean_accuracy <- mean(unlist(results))
mean_precision <- mean(precision_list)
mean_recall <- mean(recall_list)
mean_f1 <- mean(f1_list)
mean_roc_auc <- mean(roc_auc_list)

# Print results
cat("Model Performance Metrics (10-Fold Cross-Validation):\n")
cat("Accuracy: ", round(mean_accuracy, 3), "\n")
cat("Precision: ", round(mean_precision, 3), "\n")
cat("Recall: ", round(mean_recall, 3), "\n")
cat("Specificity: ", round(specificity, 3), "\n")  # Specificity metriği burada yazdırılır
cat("F1-Score: ", round(mean_f1, 3), "\n")
cat("ROC-AUC: ", round(mean_roc_auc, 3), "\n")








# CROSS-VALİDATİON OLMADAN KNN ALGORİTMASI

# Veriyi ayıralım
set.seed(123)  # Sonuçların tekrarlanabilir olması için
index <- sample(1:nrow(data_group3), nrow(data_group3)*0.8)  # %80 eğitim, %20 test

train_data <- data_group3[index, ]
test_data <- data_group3[-index, ]

# Eğitim ve test setlerinden hedef ve özellikleri ayıralım
train_x <- train_data[, -ncol(train_data)]  # Son sütun hariç tüm sütunlar
train_y <- train_data[, ncol(train_data)]   # Son sütun hedef (target)

test_x <- test_data[, -ncol(test_data)]     # Son sütun hariç tüm sütunlar
test_y <- test_data[, ncol(test_data)]      # Son sütun hedef (target)

# KNN Modeli
k <- 5  # K değeri
knn_model <- knn(train = train_x, test = test_x, cl = train_y, k = k)

# Doğruluk hesaplama
accuracy <- sum(knn_model == test_y) / length(test_y)
print(paste("Accuracy: ", accuracy))




######## Logistic Regression #######



######## Logistic Regression #######



# Korelasyon matrisini oluşturma
cor_matrix <- cor(data) 

# Korelasyon matrisini yazdırma
print(cor_matrix)

# Set a threshold for correlations
threshold <- 0.2

# Find correlations above the threshold
high_corr <- which(abs(cor_matrix) > threshold & lower.tri(cor_matrix), arr.ind = TRUE)
print(high_corr)

# Print the pairs with high correlations
correlated_pairs <- data.frame(
  Feature1 = rownames(cor_matrix)[high_corr[, 1]],
  Feature2 = colnames(cor_matrix)[high_corr[, 2]],
  Correlation = cor_matrix[high_corr]
)

print(correlated_pairs)

# # Korelasyon matrisini görselleştirme (corrplot kullanımı)
# library(corrplot)
# corrplot(correlated_pairs, method = "color", type = "upper", order = "hclust",
#          tl.col = "black", tl.srt = 45)
# 
# # Alternatif: ggcorrplot ile görselleştirme
# library(ggcorrplot)
# ggcorrplot(correlated_pairs, lab = TRUE, colors = c("red", "white", "blue"))
# 
# 
# target_corr <- correlated_pairs["basari_durumu_basarisiz", ]  # Replace "target_column" with your actual target
# target_corr <- sort(target_corr, decreasing = TRUE)
# 
# print((abs(target_corr) > 0.7))


# Set a threshold for correlations
threshold2 <- 0.2

# Filter correlations above the threshold
cor_matrix_filtered <- cor_matrix
cor_matrix_filtered[abs(cor_matrix_filtered) < threshold2] <- NA
print(cor_matrix_filtered)
# Melt the matrix for ggplot2 (long format)
cor_matrix_melt <- melt(cor_matrix_filtered, na.rm = TRUE)

# Plot using ggplot2
ggplot(cor_matrix_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), 
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Features", y = "Features", title = "Highly Correlated Features")




# Required Libraries
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
library(caret)
library(pROC)

# Dataset without highly correlated feature "destek_orani"
data_whcf <- subset(data, select = -c(destek_orani))

data_whcf_group1 <- subset(data_group1, select = -c(destek_orani))
data_whcf_group2 <- subset(data_group2, select = -c(destek_orani))
data_whcf_group3 <- subset(data_group3, select = -c(destek_orani))

# For evaluating feature selection impact on the model
data_whcf <- data_whcf_group3

# Ensure the target variable is a factor
data_whcf$basari_durumu_basarisiz <- as.factor(data_whcf$basari_durumu_basarisiz)

# Split the data into train and test sets
set.seed(123)
index <- createDataPartition(data_whcf$basari_durumu_basarisiz, p = 0.8, list = FALSE)
train_data <- data_whcf[index, ]
test_data <- data_whcf[-index, ]

# Address class imbalance with upsampling
train_data_balanced <- upSample(x = train_data[, -ncol(train_data)], 
                                y = train_data$basari_durumu_basarisiz)
colnames(train_data_balanced)[ncol(train_data_balanced)] <- "target"

# Logistic Regression Model with Cross-Validation
ctrl <- trainControl(method = "cv",        # Cross-validation method
                     number = 10,         # 10-fold CV
                     sampling = "up")     # Handle class imbalance

logistic_model <- train(target ~ ., 
                        data = train_data_balanced, 
                        method = "glm", 
                        family = "binomial", 
                        trControl = ctrl)

# Evaluate model on the test set
probabilities <- predict(logistic_model, newdata = test_data, type = "prob")[, 2]  # Olasılıklar

# Threshold Adjustment (Set Your Threshold Here)
threshold <- 0.6  # Threshold değeri burada ayarlanır
predictions <- ifelse(probabilities > threshold, "1", "0")
predictions <- factor(predictions, levels = levels(test_data$basari_durumu_basarisiz))

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Calculate Metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1 <- conf_matrix$byClass["F1"]
specificity <- conf_matrix$byClass["Specificity"]  # Specificity metriği eklendi

# ROC-AUC Calculation
roc_obj <- roc(as.numeric(test_data$basari_durumu_basarisiz), as.numeric(predictions))
roc_auc <- auc(roc_obj)

# Print Metrics
cat("Model Performance Metrics:\n")
cat("Accuracy: ", round(accuracy, 3), "\n")
cat("Precision: ", round(precision, 3), "\n")
cat("Recall: ", round(recall, 3), "\n")
cat("Specificity: ", round(specificity, 3), "\n")  # Specificity metriği burada yazdırılır
cat("F1-Score: ", round(f1, 3), "\n")
cat("ROC-AUC: ", round(roc_auc, 3), "\n")




# ----------------------------------------------------------
# ---------------Naive Bayes--------------------------------
# ----------------------------------------------------------
# Gerekli kütüphaneleri yükleyelim


library(e1071)

library(caret)  # confusionMatrix için gerekli

# Hedef değişkeni faktör (kategorik) olarak belirleyelim
data_group1$basari_durumu_basarisiz <- as.factor(data_group1$basari_durumu_basarisiz)

# Veri setini 10 katmanlı (fold) stratified cross-validation'a ayıralım
set.seed(123)  # Sonuçların tekrarlanabilir olması için
split <- vfold_cv(data_group1, v = 10, strata = "basari_durumu_basarisiz")

# Naïve Bayes modelini her fold için eğitelim
results <- list()  # Model sonuçlarını tutmak için
precision_results <- numeric(10)  # Precision sonuçlarını saklamak için
recall_results <- numeric(10)  # Recall sonuçlarını saklamak için
f1_results <- numeric(10)  # F1-Score sonuçlarını saklamak için
specificity_results <- numeric(10)  # Specificity sonuçlarını saklamak için
accuracy_results <- numeric(10)  # Doğruluk sonuçlarını saklamak için

for (i in 1:length(split$splits)) {
  # Split'ten eğitim ve test verilerini ayıralım
  train_data <- analysis(split$splits[[i]])  # Eğitim verisi
  test_data <- assessment(split$splits[[i]])  # Test verisi
  
  # Naïve Bayes modelini eğitelim
  nb_model <- naiveBayes(basari_durumu_basarisiz ~ ., data = train_data)
  
  # Test verisi ile tahmin yapalım
  predictions <- predict(nb_model, test_data)
  actual <- test_data$basari_durumu_basarisiz
  
  # Confusion matrix hesaplayalım
  cm <- confusionMatrix(predictions, actual)
  
  # Her fold için precision, recall, f1-score ve specificity'yi alalım
  precision_results[i] <- cm$byClass["Precision"]
  recall_results[i] <- cm$byClass["Recall"]
  f1_results[i] <- cm$byClass["F1"]
  specificity_results[i] <- cm$byClass["Specificity"]
  
  # Accuracy'yi de ekleyelim
  accuracy_results[i] <- cm$overall["Accuracy"]
  
  # Sonuçları kaydedelim
  results[[i]] <- cm$byClass  # Her fold için metrikleri kaydedelim
}

# Fold'lar arası ortalama doğruluğu yazdıralım
mean_accuracy <- mean(accuracy_results)
cat("Ortalama Doğruluk: ", round(mean_accuracy, 3), "\n")

# Fold'lar arası ortalama Precision, Recall, F1-Score, Specificity hesaplayalım
mean_precision <- mean(precision_results)
mean_recall <- mean(recall_results)
mean_f1 <- mean(f1_results)
mean_specificity <- mean(specificity_results)

cat("Ortalama Precision: ", round(mean_precision, 3), "\n")
cat("Ortalama Recall: ", round(mean_recall, 3), "\n")
cat("Ortalama F1-Score: ", round(mean_f1, 3), "\n")
cat("Ortalama Specificity: ", round(mean_specificity, 3), "\n")






# --------------------------------------------------------
# -----------------Neural Network-------------------------
# --------------------------------------------------------
# Gerekli kütüphaneler
library(neuralnet)
library(caret)  # confusionMatrix için

# Veri setini inceleyelim
View(data_group1)

# Hedef değişkeni belirle (örneğin: "basari_durumu_basarisiz")
target_column <- "basari_durumu_basarisiz"

# Eğitim (%70) ve test (%30) setlerine ayırma
set.seed(123)  # Tekrarlanabilirlik için
trainIndex <- createDataPartition(data_group1$basari_durumu_basarisiz, p = 0.7, list = FALSE)
train_data <- data_group1[trainIndex, ]
test_data <- data_group1[-trainIndex, ]

# 1. Hedef değişkeni binarize etme (eğer gerekliyse) ve 0-1 formatına dönüştürme
train_data$basari_durumu_basarisiz <- as.numeric(train_data$basari_durumu_basarisiz) - 1
test_data$basari_durumu_basarisiz <- as.numeric(test_data$basari_durumu_basarisiz) - 1

# 2. Sütun adlarını formüle uygun hale getirme
colnames(data_group1) <- make.names(colnames(data_group1))

# 3. Özelliklerin ismini alarak formül oluşturma
feature_names <- colnames(data_group1)[colnames(data_group1) != "basari_durumu_basarisiz"]

# Formülü oluşturma
formula <- as.formula(paste("basari_durumu_basarisiz ~", paste(feature_names, collapse = " + ")))

# 4. Neural Network Modeli -------------------------------------------------
nn_model <- neuralnet(
  formula,
  data = train_data,
  hidden = c(8, 4),  # Gizli katman: birinde 8, diğerinde 4 nöron
  linear.output = FALSE  # Sınıflandırma problemi için
)

# Model sonuçlarını incele
print(nn_model$result.matrix)
plot(nn_model)

# 5. Performans Değerlendirmesi --------------------------------------------
# Test seti tahmini
test_predictions <- neuralnet::compute(nn_model, test_data[, feature_names])$net.result
test_predictions <- ifelse(test_predictions > 0.5, 1, 0)  # Tahminleri binary sınıfa çevir

# Doğruluk hesaplama
actual <- test_data[[target_column]]
accuracy <- mean(test_predictions == actual)
cat("Test Seti Doğruluk Oranı:", round(accuracy, 3), "\n")

# Confusion matrix
cm <- confusionMatrix(as.factor(test_predictions), as.factor(actual))

# Precision, Recall, F1-Score, Specificity
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Recall"]
f1_score <- cm$byClass["F1"]
specificity <- cm$byClass["Specificity"]

cat("Precision: ", round(precision, 3), "\n")
cat("Recall: ", round(recall, 3), "\n")
cat("F1-Score: ", round(f1_score, 3), "\n")
cat("Specificity: ", round(specificity, 3), "\n")

# Hyperparameter Tuning -------------------------------------------------
# Hyperparameter grid
hidden_layers <- list(c(4, 2), c(6, 4), c(8, 4), c(10, 6))
thresholds <- c(0.01, 0.005)
grid_results <- data.frame(hidden = character(), threshold = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)

for (hidden in hidden_layers) {
  for (thresh in thresholds) {
    model <- neuralnet(
      formula,
      data = train_data,
      hidden = hidden,
      threshold = thresh,
      linear.output = FALSE
    )
    predictions <- neuralnet::compute(model, test_data[, feature_names])$net.result
    predictions <- ifelse(predictions > 0.5, 1, 0)
    acc <- mean(predictions == actual)
    grid_results <- rbind(grid_results, data.frame(hidden = I(list(hidden)), threshold = thresh, accuracy = acc))
  }
}

# En iyi sonucu yazdır
best_config <- grid_results[which.max(grid_results$accuracy), ]
cat("En İyi Konfigürasyon:\n")
print(best_config)

# En iyi modeli oluşturma
best_hidden <- eval(parse(text = best_config$hidden))  # Liste öğesini alıyoruz
best_threshold <- best_config$threshold

final_model <- neuralnet(
  formula,
  data = train_data,
  hidden = best_hidden,
  threshold = best_threshold,
  linear.output = FALSE
)

# Final model tahmini
final_predictions <- neuralnet::compute(final_model, test_data[, feature_names])$net.result
final_predictions <- ifelse(final_predictions > 0.5, 1, 0)
final_accuracy <- mean(final_predictions == actual)
cat("Final Model Doğruluk Oranı:", round(final_accuracy, 3), "\n")

#---------------------------SVM Implementation------------------------------

# Install and load the necessary packages
#install.packages("caret")
#install.packages("e1071")
#install.packages("kernlab")

library(caret)
library(e1071)
library(kernlab)

# Grid Search is commented for faster execution

# Using data group 2
data_whcf_group2 <- data_group2[, !(names(data_group2) %in% c("destek_orani"))]

# Target as factor
data_whcf_group2$basari_durumu_basarisiz <- as.factor(data_whcf_group2$basari_durumu_basarisiz)

# Define the training control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Split the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(data_whcf_group2), 0.7 * nrow(data_whcf_group2))
train_data <- data_whcf_group2[train_indices, ]
test_data <- data_whcf_group2[-train_indices, ]

#-------------------Radial Kernel------------------------------

# Define the hyperparameter grid for cost and gamma
tune_grid <- expand.grid(
  .C = 10^(-1:2),  # Range of values for cost (C)
  .sigma = c(0.5, 1, 2)  # Range of values for sigma
)

# Train the SVM model using cross-validation and grid search for hyperparameters
svm_radial_model <- train(basari_durumu_basarisiz ~ ., data = train_data, 
                          method = "svmRadial",  # Radial kernel SVM
                          trControl = train_control,  # Cross-validation settings
                          #tuneGrid = tune_grid # Hyperparameter grid for tuning
)  

# Use the best model to make predictions on the test set
predictions <- predict(svm_radial_model, newdata = test_data)

# Confusion matrix to evaluate the model's performance
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
print("SVM model Using Radial Kernel")
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#-------------------Polynomial Kernel------------------------------

# Define grid for Polynomial kernel
tune_grid_poly <- expand.grid(
  .C = 10^(-2:2),  # Cost parameter
  .degree = 2:5,  # Degree of polynomial
  .scale = c(0.1, 0.5, 1)  # Scale factor
)

# Train SVM with Polynomial kernel
svm_poly_model <- train(
  basari_durumu_basarisiz ~ ., data = train_data,
  method = "svmPoly",
  trControl = train_control,
  #tuneGrid = tune_grid_poly
)

# Use the best model to make predictions on the test set
predictions <- predict(svm_poly_model, newdata = test_data)

# Confusion matrix to evaluate the model's performance
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
print("SVM model Using Polynomial Kernel")
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#-------------------Linear Kernel------------------------------

# Define grid for Linear kernel
tune_grid_linear <- expand.grid(
  .C = 10^(-2:2)  # Cost parameter
)

# Train SVM with Linear kernel
svm_linear_model <- train(
  basari_durumu_basarisiz ~ ., data = train_data,
  method = "svmLinear",
  trControl = train_control,
  #tuneGrid = tune_grid_linear
)

# Use the best model to make predictions on the test set
predictions <- predict(svm_linear_model, newdata = test_data)

# Confusion matrix to evaluate the model's performance
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
print("SVM Model Using Linear Kernel")
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#-------------------Sigmoid Kernel------------------------------

# Train the SVM model with a sigmoid kernel
svm_sigmoid_model <- svm(
  basari_durumu_basarisiz ~ ., 
  data = train_data,
  kernel = "sigmoid",  # Use sigmoid kernel
  cost = 1,            # Example value for cost
  gamma = 0.1,         # Example value for gamma
  scale = TRUE         # Scale the data
)

# Use the best model to make predictions on the test set
predictions <- predict(svm_sigmoid_model, newdata = test_data)

# Confusion matrix to evaluate the model's performance
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
print("SVM Model Using Sigmoid Kernel")
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#-------------------Final Model------------------------------

# Out of four kernels linear had the best metrics
# Train the final model using linear kernel and whole dataset

svm_final_model <- train(
  basari_durumu_basarisiz ~ ., data = data_whcf_group2,
  method = "svmLinear",
  trControl = train_control,
  #tuneGrid = svm_linear_model$bestTune
)                   
# Use the best model to make predictions on the test set
predictions <- predict(svm_final_model, newdata = test_data)

# Confusion matrix to evaluate the model's performance
conf_matrix <- confusionMatrix(predictions, test_data$basari_durumu_basarisiz)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
print("Final Model Using Linear Kernel")
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#---------------------------Random Forest Implementation------------------------------

# install.packages("caret")
# install.packages("randomForest")
# install.packages("ranger")
# install.packages("pROC")
# install.packages("ROCR")
library(pROC)
library(ROCR)
library(ranger)
library(caret)
library(randomForest)

# Grid Search is commented for faster execution

# Using data group 2
data_whcf_group2 <- data_group2[, !(names(data_group2) %in% c("destek_orani"))]

# Using data 
data <- data[, !(names(data) %in% c("destek_orani"))]

# Target as factor
data_whcf_group2$basari_durumu_basarisiz <- as.factor(data_whcf_group2$basari_durumu_basarisiz)
data$basari_durumu_basarisiz <- as.factor(data$basari_durumu_basarisiz)


# Update levels with valid names
levels(data_whcf_group2$basari_durumu_basarisiz) <- c("Success", "Failure")
levels(data$basari_durumu_basarisiz) <- c("Success", "Failure")


# Split the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(data_whcf_group2), 0.7 * nrow(data_whcf_group2))
train_data <- data_whcf_group2[train_indices, ]
test_data <- data_whcf_group2[-train_indices, ]

num_features <- ncol(train_data) - 1  # Exclude target column

# Define the grid of hyperparameters
tune_grid <- expand.grid(
  mtry = 1:num_features ,                  # All mtry values between 1 and 32
  splitrule = "gini",             # Split rule (for classification)
  min.node.size = c(1, 5, 10)     # Nodesize values
)

# Set up cross-validation (10-fold)
train_control <- trainControl(
  method = "cv",                # Cross-validation method
  number = 10,                  # 10-fold cross-validation
  classProbs = TRUE,            # Get class probabilities for ROC curve
  summaryFunction = twoClassSummary,  # To get metrics like ROC, Sensitivity, etc.
  savePredictions = "all"       # Save all predictions for later analysis
)

results <- list()

# Loop through different ntree values
for (ntree_value in c(50,100)) {
  set.seed(123)
  model <- train(
    basari_durumu_basarisiz ~ ., 
    data = train_data, 
    method = "ranger",              # Use ranger instead of randomForest
    trControl = train_control,      # Cross-validation setup
    #tuneGrid = tune_grid,           # Grid of mtry and nodesize
    num.trees = ntree_value,        # ntree values in for loop
    metric = "ROC"                  # Use ROC as the metric for model selection
    
  )
  
  # Store the model and results
  results[[paste0("ntree_", ntree_value)]] <- model
}
# Extract the predictions (class probabilities)
predictions <- model$pred

# Extract the probabilities for the positive class
probabilities <- predictions[, "Success"]  # Change "Success" to the positive class name

# Extract the true class labels
true_labels <- factor(predictions$obs, levels = levels(train_data$basari_durumu_basarisiz))

# Plot the ROC curve
roc_curve <- roc(true_labels, probabilities)
plot(roc_curve, main = "ROC Curve", col = "#8D31BF", lwd = 2)  # Plot ROC

# Find the best model across ntree values  (best ntree value is 100)
best_model <- results[[which.min(sapply(results, function(x) min(x$results$Accuracy)))]]

#print (results)
#print(best_model)

# Best hyperparameters
print(best_model$bestTune)

# Make predictions on the test data
predictions <- predict(model, test_data)

# Generate a confusion matrix
conf_matrix <- confusionMatrix(
  factor(predictions, levels = levels(test_data$basari_durumu_basarisiz)), 
  factor(test_data$basari_durumu_basarisiz)
)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
cat("Model on test set","\n"," Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")

#-------------------------------------------------------------
# Accuracy or performance metrics
options(max.print = 1e7)  # Increasing console print limit
#print(best_model$results)

# Building the final model with best parameters
set.seed(123)
final_model <- train(
  basari_durumu_basarisiz ~ ., 
  data = data_whcf_group2,         # Can change the dataset
  method = "ranger",               # Use ranger instead of randomForest
  trControl = train_control,       # Cross-validation setup
  tuneGrid = best_model$bestTune,  # Grid of mtry and nodesize
  num.trees = 50                   # ntree value
)

# Make predictions on the test data
predictions <- predict(final_model, test_data)

# Generate a confusion matrix
conf_matrix <- confusionMatrix(
  factor(predictions, levels = levels(test_data$basari_durumu_basarisiz)), 
  factor(test_data$basari_durumu_basarisiz)
)

# Extract metrics
accuracy <- conf_matrix$overall["Accuracy"]  # Accuracy
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]       # Recall (Sensitivity)
f1_score <- 2 * ((precision * recall) / (precision + recall))  # F1-score

# Print the results
cat(" Accuracy:", accuracy, "\n","Precision:", precision, "\n","Recall:", recall, "\n","F1 Score:", f1_score, "\n")
