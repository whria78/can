#https://github.com/whria78/can
#Han Seung Seog (whria78@gmail.com)
#https://modelderm.com

def main():
    import os
    import torch
    from torchvision import datasets, transforms
    
    import numpy as np
    import argparse
    import time
    import timm

    #parse arguments
    parser = argparse.ArgumentParser(description='An example of CNN training and deployment; Han Seung Seog')
    parser.add_argument('--model', type=str, default='mobilenet', help='mobilenet / efficientnet / vgg (mobilenet by default)')
    parser.add_argument('--resolution', type=int, default=224, help='image resolution (224 by default)')

    parser.add_argument('--epoch', type=int, default=6, help='number of epochs to train (6 by default)')
    parser.add_argument('--batch', type=int, default=32, help='batch size (32 by default)')

    parser.add_argument('--opt', type=str, default='radam', help='optimizer radam / adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (0.001 by default)')

    parser.add_argument('--train', type=str, default='dataset/train', help='training image folder (/dataset/train by default)')
    parser.add_argument('--val', type=str, default='', help='validation image folder (/dataset/val by default)')
    parser.add_argument('--test', type=str, default='dataset/test', help='test image folder (/dataset/test by default)')

    parser.add_argument('--result', type=str, default='log.csv', help='save result; csv file')
    parser.add_argument('--profile', type=str, default='', help='test profile memo')

    args = parser.parse_args()

    #define transformations
    dataset_means=(0.485, 0.456, 0.406) #precomputed channel means of ImageNet(train) for normalization
    dataset_stds=(0.229, 0.224, 0.225) #precomputed standard deviations

    transformations = {
        'train': transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_means, dataset_stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_means, dataset_stds)
        ])
    }

    #load datasets and define loaders
    imagedir_train_org=args.train.split(';')
    imagedir_val=os.path.join(os.getcwd(),args.val)
    imagedir_test=os.path.join(os.getcwd(),args.test)

    f_log=open(args.result,'a')
    f_log.write(f"{args.profile},{args.opt},{args.epoch},{args.lr},{args.batch},")

    def print_dataset_info(dataset_):
        _, list_dx_count_list = torch.unique(torch.tensor(dataset_.targets), return_counts=True)
        list_dx_count_list=list_dx_count_list.to("cpu").numpy().tolist()
        for class_no,class_name in enumerate(dataset_.class_to_idx):
            print(class_name,list_dx_count_list[class_no], 'images')
            #f_log.write(f"{class_name},{list_dx_count_list[class_no]},")

    train_dataset=None
    train_dataset_list=[]
    for imagedir_train_org_ in imagedir_train_org:
        if args.test!=imagedir_train_org_:
            imagedir_train=os.path.join(os.getcwd(),imagedir_train_org_)
            dataset_=datasets.ImageFolder(imagedir_train,transformations['train'])
            print(imagedir_train,len(dataset_))
            train_dataset_list+=[dataset_]
            
        else:
            print("SKIP :",imagedir_train_org_)

    train_dataset=torch.utils.data.ConcatDataset(train_dataset_list)    
    print("Train : ",imagedir_train,len(train_dataset))
    f_log.write(f"{len(train_dataset)},")

    val_dataset=None
    if args.val!="":    
        val_dataset=datasets.ImageFolder(imagedir_val,transformations['val'])
        print("Validation : ",imagedir_val)
        print_dataset_info(val_dataset)
    test_dataset=datasets.ImageFolder(imagedir_test,transformations['val'])
    print("Test : ",imagedir_test)
    print_dataset_info(test_dataset)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True,num_workers=6)
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,num_workers=4)

    len_classes=len(test_dataset.class_to_idx)


    #choose model to train
    if args.model == 'mobilenet':
        model=timm.create_model('mobilenetv2_100',num_classes=len_classes,pretrained=True)
    elif args.model == 'efficientnet':
        model=timm.create_model('efficientnet_lite0',num_classes=len_classes,pretrained=True)
    elif args.model == 'vgg':
        model=timm.create_model('vgg19_bn',num_classes=len_classes,pretrained=True)
    else:
        print("ERR - No dataset")

    #number of iterations
    n_epochs = args.epoch
    #optimizer and scheduler
    if args.opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt=='radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)

    #loss function (standard cross-entropy taking logits as inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    #train on GPU if CUDA is available, else on CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    #print training information
    print("")
    if torch.cuda.is_available():
        hardware = "GPU " + str(device) 
    else:
        hardware = "CPU (CUDA was not found)" 
    print("Training information:")
    print("hardware:", hardware)
    print("total number of epochs:", n_epochs)
    print("mini batch size:", args.batch)
    print("")


    #training loop
    lowest_val_loss = np.inf #used for saving the best model with lowest validation loss
    for epoch in range(1, n_epochs+1):
        #train model
        start_time = time.time()
        train_losses = []
        model.train()
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.shape[0]
            print("train mini batch " + str(i+1) + "/" + str(len(train_loader)) + " - %d training images processed" % (i*batch_size), end="\r", flush=True) 
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        print("                                                                         ", end="\r", flush=True) #delete output from train counter to not interfere with validation counter (probably can be done better)

        #validate model
        val_accuracy=0
        val_loss=0
        if val_loader is not None:
               
            with torch.no_grad():
                model.eval()
                correct_labels = 0
                all_labels = 0
                val_losses = []
                for i, (imgs, labels) in enumerate(val_loader):
                    print("valid batch " + str(i+1) + "/" + str(len(val_loader)), end="\r", flush=True) 
                    imgs, labels = imgs.to(device), labels.to(device)
                    batch_size = imgs.shape[0]
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    val_losses.append(loss.item())
                    _, preds = torch.max(outputs, dim=1) #predictions
                    matched = preds == labels #comparison with ground truth
                    
                    correct_labels += float(torch.sum(matched)) 
                    all_labels += float(batch_size) 

                val_accuracy = correct_labels / all_labels #compute top-1 accuracy on validation data 
    
            val_loss = np.mean(val_losses)
        
        train_loss = np.mean(train_losses)
        
        end_time = time.time()
        
        #print iteration results
        print("Epoch: %d/%d, lr: %f, train_loss: %f, val_loss: %f, val_acc: %f, time(sec): %f" % (epoch, n_epochs, optimizer.param_groups[0]['lr'], train_loss, val_loss, val_accuracy, end_time - start_time))

    #torch.save(model.state_dict(), 'prev.pth')
    #print("Save Pytorch model")
    print("Finish Training")

    #TEST
    start_time = time.time()

    all_label_list=[] #for auroc
    all_output_list=[] #for auroc

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference   

    with torch.no_grad():
        model.eval()
        correct_labels = 0
        all_labels = 0
        val_losses = []
        for i, (imgs, labels) in enumerate(test_loader):
            print("test batch " + str(i+1) + "/" + str(len(test_loader)), end="\r", flush=True) 
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.shape[0]
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            val_losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1) #predictions
            matched = preds == labels #comparison with ground truth
            correct_labels += float(torch.sum(matched))
            all_labels += float(batch_size)
            
            all_label_list+=labels.to("cpu").numpy().tolist()
            all_output_list+=outputs.to("cpu").numpy().tolist()

        test_accuracy = correct_labels / all_labels #compute top-1 accuracy on validation data 
        end_time = time.time()
        print("TEST Accuracy: %f, time(sec): %f" % (test_accuracy, end_time - start_time))

    #Calculate AUC save xls for R statistics
    from sklearn.metrics import roc_auc_score
    from openpyxl import Workbook
    
    for class_no,class_name in enumerate(test_dataset.class_to_idx):
        print(class_name)
        y_real=[]
        y_pred=[]
        for no_,all_output_list_ in enumerate(all_output_list): 
            all_output_list_softmax_=softmax(all_output_list_)
            if  all_label_list[no_]==class_no:
                y_real+=[1]
            else:
                y_real+=[0]
            y_pred+=[all_output_list_softmax_[class_no]]
  
        score = roc_auc_score(np.array(y_real), np.array(y_pred))
        print(f"ROC AUC: {score:.4f}")
        if class_no==0:
            f_log.write(f"{score:.4f},{args.train},{args.test}\n")

        wb = Workbook()    
        sheet1 = wb.active
        sheet1.title = class_name
        sheet1.cell(row=1, column=1).value = "dx"
        sheet1.cell(row=1, column=2).value = "pred"
        for no_,all_output_list_ in enumerate(all_output_list): 
            all_output_list_softmax_=softmax(all_output_list_)
            sheet1.cell(row=(no_+2), column=1).value = (all_label_list[no_]==class_no)
            sheet1.cell(row=(no_+2), column=2).value = all_output_list_softmax_[class_no]

        save_path=os.path.join(os.getcwd(),'stat','roc_%s.xlsx' % (class_name))
        try:os.makedirs(os.path.dirname(save_path))
        except:pass
        wb.save(filename=save_path)
        excel_path=save_path

        f=open(os.path.join(os.getcwd(),'template','r.template'),'r')
        template_list=f.readlines()
        f.close()
        template_=''
        for t_ in template_list:
            template_+=t_+'\n'
        
        template_=template_.replace('[XLS_PATH]',excel_path).replace('[TEST_NO_IMAGES]','%d' % (len(test_dataset))).replace('\\','/')
        
        save_path=os.path.join(os.getcwd(),'stat','R_%s.txt' % (class_name))
        f=open(save_path,'w')
        f.write(template_)
        f.close()
        
        
if __name__ == '__main__':
    main()
