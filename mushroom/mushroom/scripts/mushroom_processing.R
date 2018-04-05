
source("/Users/berk/Desktop/Dropbox/Research/SLIM/R/DataProcessingFunctions.R")

#Load Dataset
data_name 				= "mushroom"
raw_data_dir  		= setwd(paste0("/Users/berk/Desktop/Dropbox/Research/SLIM/Data/Raw Data Files/",data_name));
raw_data_file 		= "agaricus-lepiota.data"
csv_file_name  		= paste0(data_name,"_processed.csv")
csv_bin_file_name = paste0(data_name,"_binary_processed.csv")


##### Load Data ##### 
#http://archive.ics.uci.edu/ml/datasets/Mushroom
data = read.csv(raw_data_file,header=FALSE)

#Place Outcome at End
data = data[,c(names(data)[2:length(names(data))],"V1")]

##### Rename Features ##### 

# 1. cap-shape:                	bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface:              	fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color:               	brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?:                 	bruises=t,no=f
# 5. odor:                     	almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment:          	attached=a,descending=d,free=f,notched=n
# 7. gill-spacing:             	close=c,crowded=w,distant=d
# 8. gill-size:                	broad=b,narrow=n
# 9. gill-color:               	black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 10. stalk-shape:              enlarging=e,tapering=t
# 11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 16. veil-type:                partial=p,universal=u
# 17. veil-color:               brown=n,orange=o,white=w,yellow=y
# 18. ring-number:              none=n,one=o,two=t
# 19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 21. population:               abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# 
names(data) = c("cap_shape","cap_surface","cap_color",
								"bruises","odor",
								"gill_attachment","gill_spacing","gill_size","gill_color",
								"stalk_shape","stalk_root",
								"stalk_surface_above_ring","stalk_surface_below_ring",
								"stalk_color_above_ring","stalk_color_below_ring",
								"veil_type","veil_color",
								"ring_number","ring_type",
								"spore_print_color","population","habitat","poisonous")

##### Rename Levels ##### 
# 1. cap-shape:                	bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
levels(data$cap_shape) 				= c("bell","conical","flat","knobbed","sunken","convex")

# 2. cap-surface:              	fibrous=f,grooves=g,smooth=s,scaly=y,
levels(data$cap_surface) 			= c("fibrous","grooves","smooth","scaly")

# 3. cap-color:               	buff=b,cinnamon=c,red=e,gray=g,brown=n,pink=p,green=r,purple=u,white=w,yellow=y
levels(data$cap_color) 				= c("buff","cinnamon","red","gray","brown","pink","green","purple","white","yellow")

# 4. bruises?:                 	bruises=t,no=f
levels(data$bruises) 					= c(FALSE,TRUE)

# 5. odor:                     	almond=a,creosote=c,foul=f,anise=l,musty=m,none=n,pungent=p,spicy=s,fishy=y
#"a" "c" "f" "l" "m" "n" "p" "s" "y"
levels(data$odor) 						 = c("almond","creosote","foul","anise","musty","none","pungent","spicy","fishy")

# 6. gill-attachment:          	attached=a,descending=d,free=f,notched=n #CANNOT FIND: NOTCHED/DESCENDING?
levels(data$gill_attachment) 		= c("attached","free")

# 7. gill-spacing:             	close=c,crowded=w,distant=d
levels(data$gill_spacing) 			= c("close","crowded")

# 8. gill-size:                	broad=b,narrow=n
levels(data$gill_size) 				= c("broad","narrow")

# 9. gill-color:               	buff=b,red=e,gray=g,chocolate=h,black=k,brown=n,orange=o,pink=p,green=r,purple=u,white=w,yellow=y
# "b" "e" "g" "h" "k" "n" "o" "p" "r" "u" "w" "y"
levels(data$gill_color) 				= c("buff","red","gray","chocolate","black","brown","orange","pink","green","purple","white","yellow")

# 10. stalk-shape:              enlarging=e,tapering=t
levels(data$stalk_shape) 				= c("elarging","tapering")

# 11. stalk-root:               missing=?, bulbous=b,club=c,equal=e,rooted=r,  CANNOT FIND: cup=u,rhizomorphs=z
levels(data$stalk_root) 				= c("missing","bulbous","club","equal","rooted");

# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
levels(data$stalk_surface_above_ring) = c("fibrous","grooves","smooth","scaly")

# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
levels(data$stalk_surface_below_ring) = c("fibrous","grooves","smooth","scaly")

# 14. stalk-color-above-ring:   buff=b,cinnamon=c,red=e,gray=g,brown=n,orange=o, pink=p,white=w,yellow=y
levels(data$stalk_color_above_ring) = c("buff","cinnamon","red","gray","brown","orange", "pink","white","yellow")

# 15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
levels(data$stalk_color_below_ring) = c("buff","cinnamon","red","gray","brown","orange", "pink","white","yellow")

# 16. veil-type:                partial=p, CANNOT FIND:universal=u 
levels(data$veil_type)  				= c("partial")

# 17. veil-color:               brown=n,orange=o,white=w,yellow=y
levels(data$veil_color)  				= c("brown","orange", "white","yellow")

# 18. ring-number:              none=n,one=o,two=t
levels(data$ring_number)				= c(0,1,2)

# 19. ring-type:                evanescent=e,flaring=f,large=l,none=n,pendant=p, CANNOT FIND:: cobwebby=c,sheathing=s,zone=z
levels(data$ring_type)					= c("evanescent","flaring", "large","none","pendant")

# 20. spore-print-color:        buff=b,chocolate=h,black=k,brown=n,orange=o,green=r,purple=u,white=w,yellow=y
levels(data$spore_print_color)	= c("buff","chocolate","black","brown","orange","green","purple","white","yellow")

# 21. population:               abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
levels(data$population) 				= c("abundant","clustered","numerous","scattered","several","solitary")
	
# 22. habitat:                  woods=d,grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,
#"d" "g" "l" "m" "p" "u" "w"
levels(data$habitat) 						= c("woods","grasses","leaves","meadows","paths","urban","waste")

#OUTCOME: poisonous (e= edible,b=poisonous)
levels(data$poisonous) 					= c(-1,1)
data$poisonous = as.numeric(data$poisonous)
neg_ind = data$poisonous==1
pos_ind = data$poisonous==2
data$poisonous[neg_ind] = -1
data$poisonous[pos_ind] = 1

##### Additional Processing ######

#Remove Categorical Features
varlist = setdiff(names(data),"poisonous")
data = binarize.categorical.variables(df=data,varlist=varlist,remove_categorical_variable = TRUE)

#Place Outcome at End
data = data[,c(setdiff(names(data),"poisonous"),"poisonous")]

#Drop Useless Features

#cannot use missing
data$stalk_root_eq_missing<-NULL

#would rather model tells us that it bruises
data$bruises_eq_FALSE<-NULL
data$odor_eq_none
data$ring_number_eq_0<-NULL

#remove any variables that are the same throughout
data = remove.variables.without.variance(data)

##### Write Full Version

write.csv(x=data,file=csv_file_name,row.names=FALSE,quote=FALSE)

##### Write Version for Rule Based Models
bdata = remove.complements.of.variables(data)
write.csv(x=bdata,file=csv_bin_file_name,row.names=FALSE,quote=FALSE)



