function f=Optimizer(X,Y, mad_values, model)
        %X=table2array(X);
        %Y=table2array(Y);
        alpha=1.5;
        %X=percDiff(X,Y);
        l1_norm = norm(minus(Y',X')/mad_values);
        dist=l1_norm;

        scores = model(X);
        prime_pred = scores';      

        %if prime_pred(1,2) > prime_pred(1,1)   %prime_pred(1,2) represents probability of belonging to class 0
        if prime_pred(1,2) >= 0.5
            cost= alpha*(1-prime_pred(1,2)) + dist;
        else
            cost= alpha*(1-prime_pred(1,2)) + dist + 100;
        end

        f=cost;
end


