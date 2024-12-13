FROM node:20.15.0

WORKDIR /app

# Copy project
COPY . .

# Install dependencies
RUN npm install

# Expose port 3000
EXPOSE 3000

# Command to run the frontend server
CMD ["npm", "run", "dev"]