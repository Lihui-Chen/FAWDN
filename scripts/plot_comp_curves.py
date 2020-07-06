import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    path_1 = '/home/ser606/clh/AIM/experiments/ADAWDFBN_trained_on_mixdataset/records/train_records.csv'
    path_2 = '/home/ser606/clh/AIM/experiments/ADAWDFBN_in3f16_x4_only_MedImage/records/train_records.csv'
    # path_3 = '/media/ser606/Elements SE/scientific_data/AIM_results/AIM/experiments/SRDENSENETFB_2W(ADAWDFBN_x4)/records/train_records.csv'
    # path_4 = '/media/ser606/Elements SE/scientific_data/srdensenet_w_attention/experiments/SRDENSENETFB_2W/records/train_records.csv'

    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    # df3 = pd.read_csv(path_3)
    # df4 = pd.read_csv(path_4)
    # psnr0=df0['psnr']
    psnr1=df1['psnr'][:200]
    psnr2=df2['psnr'][:200]
    # psnr3=df3['psnr'][:200]
    # psnr4=df4['psnr']

    valid_loss1 = df1['val_loss'][:200]
    valid_loss2 = df2['val_loss'][:200]

    train_loss1 = df1['train_loss'][:200]
    train_loss2 = df2['train_loss'][:200]

    xrange = np.arange(0, 200)


    plt.figure()

    # labels= 'G2R1T0', 'G2R2T0', 'G2
    # plt.plot(psnr0, color='skyblue', label='G2R2T0')

    plt.plot(xrange, psnr1, color ='r', label='Mix-Dataset')
    plt.plot(xrange, psnr2, color='b',label='Medical Dataset')
    # plt.plot(xrange, psnr3, color='g', label='DIV2K')
    # plt.plot(psnr4, color='r', label='FAWDN')

    #     p2.plot(psnr1, color='g', label='G2R1T0')
    #     p2.plot(psnr2, color='r', label='G2R2T1')
    #     p2.plot(psnr3, color='b', label='G2R3T2')
    plt.ylim()
    plt.legend(loc=4)
    plt.ylabel("PSNR(dB)")
    plt.xlabel("Epoch")
    plt.title('PSNR curves on the valid dataset')


    ##  zoom-in the curve
    # tx0=100
    # tx1=150
    # ty0=31.5
    # ty1=32.3

    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # plt.plot(sx,sy,"gray")
    # plt.axis([31, 32, 100, 150])

    plt.savefig('curve_of_vaild_psnr.eps', format='eps')
    plt.show()

    plt.figure()
    plt.plot(xrange, valid_loss1, color='r', label='Mix-Dataset')
    plt.plot(xrange, valid_loss2, color='b',label='Medical Dataset')

    plt.legend()
    plt.ylabel("vaild loss")
    plt.xlabel("Epoch")
    plt.title('Vaild loss curves on the valid dataset')

    plt.savefig('curve_of_val_loss.eps', format='eps')
    plt.show()

    plt.figure()
    plt.plot(xrange, train_loss1, color='r', label='Mix-Dataset')
    plt.plot(xrange, train_loss2, color='b', label='Medical Dataset')

    plt.legend()
    plt.ylabel("train loss")
    plt.xlabel("Epoch")
    plt.title('Train loss curves on train datasets')

    plt.savefig('curve_of_train_loss.eps', format='eps')
    plt.show()

    #######

    # plt.plot(psnr1, color='r')
    # plt.plot(psnr2, color='b')
    # plt.plot(psnr3, color='g')
    # plt.plot(psnr4, color='r')
    # plt.axis([tx0, tx1, ty0, ty1])
    # plt.savefig('chop_ives_adaw.eps', format='eps')

    # plt.show()
    #

if __name__ == "__main__":
    main()