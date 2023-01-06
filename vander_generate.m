function V = vander_generate(X,T,w)

I = eye(size(X,1),size(X,1));
V = zeros(size(X,1)*(T+1),size(X,1)*(T+1));
m = size(X,1);
for k = 1:(T+1)
    A = I;
    for p = 1:T
        A = [A,I*w(k)^p];
    end
    V((k-1)*m+1:k*m,:) = A;
end