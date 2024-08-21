
function results=percDiff(x,y)

    %output =zeros(1, length(y));
    %x=table2array(x);

    for i=1:length(y)
        val = abs(x(i) - y(i))/(abs(y(i)) + 0.00000000001);
        sig_mo = (1/(1+ exp(-val)));

        if round(sig_mo, 2) > 0.9
            x(i) = y(i);
        end
    end
    
    results=x;
    
end