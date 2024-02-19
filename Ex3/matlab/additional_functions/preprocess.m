function processed = preprocess(images)
    % Reshape to vectors: #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    images = bsxfun(@rdivide, images, max(images));
    % subtract the mean
    processed = bsxfun(@minus, images, mean(images));
end