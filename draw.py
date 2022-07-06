import dill 
import matplotlib.pyplot as plt








if __name__ == '__main__':
    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.2", "rb") as dill_file:
        a = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.2", "rb") as dill_file:
        b = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.2", "rb") as dill_file:
        c = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.2", "rb") as dill_file:
        d = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.3epochs=1num_microbatch200local_updates=5shuffle_rate=0.2", "rb") as dill_file:
        d1 = dill.load(dill_file)


    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.5", "rb") as dill_file:
        e = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.5", "rb") as dill_file:
        f = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.5", "rb") as dill_file:
        g = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.5", "rb") as dill_file:
        h = dill.load(dill_file)


    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.8", "rb") as dill_file:
        i = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.8", "rb") as dill_file:
        j = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.03epochs=1num_microbatch200local_updates=5shuffle_rate=0.8", "rb") as dill_file:
        k = dill.load(dill_file)
    with open("Non-iid,Partition,clip&noise, l2_norm=2.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=0.8", "rb") as dill_file:
        l = dill.load(dill_file)

    with open("Non-iid,Partition,clip&noise, l2_norm=1.3noise_multiplier=0.08epochs=1num_microbatch200local_updates=5shuffle_rate=1", "rb") as dill_file:
        m = dill.load(dill_file)
    

    fig, (sub1, sub2, sub3, sub4) = plt.subplots(1, 4)
    # fig.suptitle('Horizontally stacked subplots')
    # ax1.plot(x, y)
    # ax2.plot(x, -y)

    sub1.plot(a, label="80% Similarity | weak noise | strong clipping", alpha = 0.7, color="green")
    sub1.plot(b, label="80% Similarity | strong noise | strong clipping", alpha = 0.7, color="black")
    sub1.plot(c, label="80% Similarity | weak noise | weak clipping", alpha = 0.7, color="yellow")
    sub1.plot(d, label="80% Similarity | strong noise | weak clipping", alpha = 0.7, color="pink")
    sub1.plot(d1, label="80% Similarity | strong noise | weak clipping", alpha = 0.7, color="black")
    

    sub2.plot(e, label="50% Similarity | weak noise | strong clipping", alpha = 0.7, color="cyan")
    sub2.plot(f, label="50% Similarity | strong noise | strong clipping", alpha = 0.7, color="red")
    sub2.plot(g, label="50% Similarity | weak noise | weak clipping", alpha = 0.7, color="blue")
    sub2.plot(h, label="50% Similarity | strong noise | weak clipping", alpha = 0.7, color="purple")

    sub3.plot(i, label="20% Similarity | weak noise | strong clipping", alpha = 0.7, color="cyan")
    sub3.plot(j, label="20% Similarity | strong noise | strong clipping", alpha = 0.7, color="red")
    sub3.plot(k, label="20% Similarity | weak noise | weak clipping", alpha = 0.7, color="blue")
    sub3.plot(l, label="20% Similarity | strong noise | weak clipping", alpha = 0.7, color="purple")

    sub4.plot(m, label="0% Similarity | strong noise | weak clipping", alpha = 0.7, color="purple")
    
    
    
    plt.plot()
    # plt.ylabel('training loss')
    plt.ylabel('testing accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()